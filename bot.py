import logging
from datetime import datetime, timedelta
import config
import sqlite3
import json
import os
import threading
import copy
from tox_analiser import is_toxic
from telegram import (
    ForceReply,
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Bot,
    ChatPermissions,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
    User,
)
from telegram.ext import (
    CallbackContext,
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    InlineQueryHandler,
)
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers import pipeline
from datasets import Dataset

logger = logging.getLogger(__name__)


# 1. Вынесение работы с БД в отдельный класс
class Database:
    def __init__(self, db_name: str = "_base.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id INTEGER PRIMARY KEY,
                    rules TEXT
                )"""
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    user_id INTEGER,
                    chat_id INTEGER,
                    user_name TEXT,
                    action TEXT,
                    message TEXT,
                    message_label TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def add_record_to_cache(
        self, user_id, chat_id, user_name, action, message_text, message_label
    ):
        with self.conn:
            self.conn.execute(
                "INSERT INTO cache(user_id,chat_id, user_name, action,message, message_label, timestamp) VALUES(?,?,?,?,?,?,?)",
                (user_id, chat_id, user_name, action, message_text, message_label, datetime.now()),
            )

    def flush_cache(self):
        with self.conn:
            self.conn.execute("DROP TABLE cache")

    def execute(self, query: str, params: tuple = ()):
        try:
            with self.conn:
                return self.conn.execute(query, params)
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")


db = Database()


# 2. Безопасная работа с файлами через контекстный менеджер
class FinetuningData:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({"count": 0, "data": []}, f)

    def load(self):
        with open(self.filepath, "r") as f:
            return json.load(f)

    def save(self, data):
        with open(self.filepath, "w") as f:
            json.dump(data, f)


finetuning_data = FinetuningData("train data/data for finetuning.json")


# 3. Вынесение модели в отдельный класс с обработкой ошибок
class ToxicityClassifier:
    def __init__(self):
        try:
            self.model = BertForSequenceClassification.from_pretrained(
                "results/ad/checkpoint-165"
            )
            self.tokenizer = BertTokenizer.from_pretrained("results/ad/checkpoint-165")
            self.pipe = pipeline(
                model=self.model, tokenizer=self.tokenizer, task="text-classification"
            )

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    async def classify(self, text: str) -> str:
        try:
            if is_toxic(text):
                return "insult"
            result = self.pipe(text)
            return result[0]["label"]
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "error"

    @staticmethod
    def _process_result(res: dict) -> str:
        if res["scores"][0] > 0.15:
            return res["labels"][0]
        return "neutral"


classifier = ToxicityClassifier()

mute_permission = ChatPermissions(
    can_send_messages=False,
    can_send_polls=False,
    can_send_other_messages=False,
    can_add_web_page_previews=False,
    can_change_info=False,
    can_invite_users=False,
    can_pin_messages=False,
)
unmute_permission = ChatPermissions(
    can_send_messages=True,
    # can_send_media_messages=True,
    can_send_polls=True,
    can_send_other_messages=True,
    can_add_web_page_previews=True,
    can_change_info=False,
    can_invite_users=True,
    can_pin_messages=False,
    
)


# bot = Bot(config.BOT_TOKEN)
# 4. Улучшенные обработчики команд с проверкой прав
async def _validate_admin(update: Update) -> bool:
    user = await update.effective_chat.get_member(update.effective_user.id)
    return user.status in ["administrator", "creator"]


async def insult(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate_admin(update):
        await update.message.reply_text("Требуются права администратора")
        return
    if update.message.reply_to_message:
        message = update.message.reply_to_message
        text = message.text or message.caption or ""
        await register_new_datachunk({"label": "insult", "text": text})
        await update.message.reply_text("Сообщение помечено как оскорбление")


async def ad(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate_admin(update):
        await update.message.reply_text("Требуются права администратора")
        return
    if update.message.reply_to_message:
        ad_content = update.message.reply_to_message.text
        await register_new_datachunk({"label": "advertising", "text": ad_content})
        await update.message.reply_text("registred as ad")


async def porn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate_admin(update):
        await update.message.reply_text("Требуются права администратора")
        return
    if update.message.reply_to_message:
        adult_content = update.message.reply_to_message.text
        await register_new_datachunk({"label": "obscenity", "text": adult_content})
        await update.message.reply_text("registred as obscenity")


async def politics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _validate_admin(update):
        await update.message.reply_text("Требуются права администратора")
        return
    if update.message.reply_to_message:
        politics_content = update.message.reply_to_message.text
        await register_new_datachunk({"label": "insult", "text": politics_content})
        await update.message.reply_text("registred as politics")


# 5. Безопасное добавление данных для дообучения
async def register_new_datachunk(datachunk: dict) -> None:
    try:
        data = finetuning_data.load()
        data["data"].append(datachunk)
        data["count"] += 1

        if data["count"] >= 100:
            await start_finetuning(data)
            data["count"] = 0
            data["data"] = []

        finetuning_data.save(data)
    except Exception as e:
        logger.error(f"Failed to save data: {e}")


# 6. Улучшенный механизм дообучения
async def start_finetuning(data: dict):
    def _train():
        try:
            train_model = copy.deepcopy(classifier.model)
            train_model.train()

            dataset = Dataset.from_dict(
                {
                    "text": [item["text"] for item in data["data"]],
                    "label": [config.LABEL_MAP[item["label"]] for item in data["data"]],
                }
            )

            tokenized_dataset = dataset.map(
                lambda x: classifier.tokenizer(
                    x["text"], padding="max_length", truncation=True, max_length=128
                ),
                batched=True,
            )

            training_args = TrainingArguments(
                output_dir="./finetuned",
                per_device_train_batch_size=8,
                num_train_epochs=3,
                learning_rate=1e-5,
                evaluation_strategy="no",
                save_strategy="epoch",
                logging_steps=50,
            )

            trainer = Trainer(
                model=train_model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )

            trainer.train()
            trainer.save_model()

        except Exception as e:
            logger.error(f"Training failed: {e}")

    threading.Thread(target=_train).start()


# Остальные обработчики остаются аналогичными с добавлением валидации
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(await load_chat_rules(update.message.chat.id))


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    for i in update.message.entities:
        print(i.user)
    if text.strip()[0] == "\\":
        return
    result = await classifier.classify(text)
    await update.message.reply_text(result, reply_to_message_id=update.message.id)
    if result != "neutral":
        precepts: dict[str, dict[str, int]] = (
            await load_chat_rules(update.message.chat_id)
        )[result]

        cache_record = ""
        user_id, chat_id, user_name, msg_text = (
            update.message.from_user.id,
            update.message.chat.id,
            update.message.from_user.username,
            update.message.text,
        )

        if precepts["ban"] == 1:
            # await context.bot.ban_chat_member(chat_id=update.message.chat.id, user_id=update.message.from_user.id)
            # if not await _validate_admin(update):
            #     await update.message.reply_text(
            #         "BAN", reply_to_message_id=update.message.id
            #     )
            # await update.message.reply_text(
            #     "BAN", reply_to_message_id=update.message.id
            # )
            cache_record += "ban"
            print("ban")
            if not await _validate_admin(update):
                await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        if precepts["mute"] == 1:
            cache_record += "mute"
            print(f'Мут на {precepts["muteduration"]} минут с {datetime.now()} до {datetime.now() + timedelta(minutes=int(precepts["muteduration"]))}')
            
            if not await _validate_admin(update):
                
                await context.bot.restrict_chat_member(
                    chat_id=chat_id,
                    user_id=user_id,
                    permissions=mute_permission,
                    until_date=datetime.now() + timedelta(minutes=int(precepts['muteduration'])),
                )
            # await update.message.reply_text(
            #     "MUTE", reply_to_message_id=update.message.id
            # )
        if precepts["del"] == 1:
            cache_record += "del"
            await update.message.delete()
        db.add_record_to_cache(user_id, chat_id, user_name, cache_record, msg_text, result)


# настройки нейрухи
async def load_chat_rules(chat_id) -> dict[str, dict[str, int]]:
    rules_str = db.execute(
        "SELECT rules FROM chats WHERE chat_id=?", (chat_id,)
    ).fetchone()
    if not rules_str or not rules_str[0]:
        default = "politics:ban=0,mute=0,muteduration=0,del=0; insult:ban=0,mute=0,muteduration=0,del=0; porn:ban=0,mute=0,muteduration=0,del=0; advertising:ban=0,mute=0,muteduration=0,del=0"
        db.execute(
            "INSERT OR REPLACE INTO chats (chat_id, rules) VALUES (?, ?)",
            (chat_id, default),
        )
        db.conn.commit()
        return {
            k: {i: int(j) for i, j in [n.split("=") for n in v.split(",")]}
            for k, v in [item.split(":") for item in default.split("; ")]
        }
    return {
        k: {i: int(j) for i, j in [n.split("=") for n in v.split(",")]}
        for k, v in [item.split(":") for item in rules_str[0].split("; ")]
    }


async def set_chat_rules(rules: str | dict[str, dict[str, int]], chat_id: int):
    if type(rules) == str:
        db.execute("UPDATE chats SET rules=? WHERE chat_id=?", (rules, chat_id))
        db.conn.commit()
    else:
        rules_str = "; ".join(
            [
                f"{k}:{','.join([f'{i}={j}' for i,j in v.items()])}"
                for k, v in rules.items()
            ]
        )
        db.execute("UPDATE chats SET rules=? WHERE chat_id=?", (rules_str, chat_id))
        db.conn.commit()


async def get_rules_buttons():
    buttons = [
        [InlineKeyboardButton("Politics", callback_data="rule:politics")],
        [InlineKeyboardButton("Ad", callback_data="rule:advertising")],
        [InlineKeyboardButton("Porn", callback_data="rule:porn")],
        [InlineKeyboardButton("Insult", callback_data="rule:insult")],
    ]
    return InlineKeyboardMarkup(buttons)


async def get_rule_state_buttons(chat_id: int, rule: str):
    rules = await load_chat_rules(chat_id)
    buttons = [
        [
            InlineKeyboardButton(
                ("✅" if rules[rule]["ban"] else "❌") + " Ban",
                callback_data=f"config:{rule}:ban",
            )
        ],
        [
            InlineKeyboardButton(
                ("✅" if rules[rule]["mute"] else "❌") + " Mute",
                callback_data=f"config:{rule}:mute",
            )
        ],
        # [
        #     InlineKeyboardButton(
        #         ("✅" if rules[rule]["warn"] else "❌") + " Warn",
        #         callback_data=f"config:{rule}:warn",
        #     )
        # ],
        [
            InlineKeyboardButton(
                ("✅" if rules[rule]["del"] else "❌") + " Del",
                callback_data=f"config:{rule}:del",
            )
        ],
        [InlineKeyboardButton("Go back", callback_data=f"config:back")],
    ]
    if rules[rule]["mute"]:
        buttons.insert(
            2,
            [
                InlineKeyboardButton(
                    f"Mute duration: {rules[rule]['muteduration']}",
                    callback_data=f"config:{rule}:mute:chmuteduration",
                )
            ],
        )
    return InlineKeyboardMarkup(buttons)


async def get_mute_duration_button(chat_id: int, rule: str):

    pass


async def get_chmuteduration_markup(rule: str):
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"/setmuteduration {rule} <длительность>",
                )
            ]
        ]
    )


async def set_mute_duration(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды для установки длительности мута"""
    if not await _validate_admin(update):
        await update.message.reply_text("Требуются права администратора")
        return

    try:
        # Получаем значение из команды
        rule = context.args[0]
        duration = int(context.args[1])

        # Определяем, для какого правила устанавливаем длительность
        # (можно хранить в context.user_data)

        if rule and duration:
            # Обновляем правила
            rules = await load_chat_rules(update.message.chat.id)
            rules[rule]["muteduration"] = duration
            await set_chat_rules(rules, update.message.chat.id)

            await update.message.reply_text(
                f"✅ Длительность мута для '{rule}' установлена: {duration} минут"
            )
    except (IndexError, ValueError):
        await update.message.reply_text(
            "Использование: /setmuteduration <правило> <время>"
        )


async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("query action")
    query = update.inline_query.query
    print(query)
    if query.startswith("Длительность мута:"):
        print("inside IF")
        # Извлекаем текущее значение из запроса
        current_value = query.split(":")[1].strip()

        # Создаем результаты для инлайн-режима
        results = [
            InlineQueryResultArticle(
                id="1",
                title=f"Установить: {current_value} минут",
                input_message_content=InputTextMessageContent(
                    message_text=f"/setmuteduration {current_value}"
                ),
            )
        ]
        await update.inline_query.answer(results)


async def button_click(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    data = query.data
    parts = data.split(":")
    print(parts)
    if parts[0] == "rule":
        await update.callback_query.edit_message_text(
            text="выберите настройки",
            reply_markup=await get_rule_state_buttons(
                rule=parts[1], chat_id=update._effective_chat.id
            ),
        )
    if parts[0] == "config":
        if parts[1] == "back":
            await update.callback_query.edit_message_text(
                text="выберите правило, которое хотите изменить",
                reply_markup=await get_rules_buttons(),
            )
        if len(parts) == 3:
            rules = await load_chat_rules(update._effective_chat.id)
            rules[parts[1]][parts[2]] = int(not rules[parts[1]][parts[2]])
            await set_chat_rules(rules, chat_id=update._effective_chat.id)
            await update.callback_query.edit_message_text(
                text="Выберите настройки",
                reply_markup=await get_rule_state_buttons(
                    rule=parts[1], chat_id=update._effective_chat.id
                ),
            )
        if len(parts) == 4:
            await update._effective_chat.send_message(
                text=f"Введите следующую команду:\n `/setmuteduration {parts[1]}` *<длительность>* \n Вместо *<длительность>* напишите желаемое время мута",
                parse_mode="Markdown",
            )


async def set_rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("choose", reply_markup=await get_rules_buttons())



async def appellate(update: Update, context: CallbackContext) -> None:
    """
    Обрабатывает команду /appelate в двух режимах:
    1. Без аргументов - откатывает действия для отправителя команды
    2. С указанием username - откатывает действия для указанного пользователя
    """
    # Проверка наличия прав у отправителя
    if not await is_user_admin(update, context):
        await update.message.reply_text("❌ Требуются права администратора!")
        return

    target_user = None

    if context.args:
        # Режим с указанием username
        try:
            username = context.args[0].strip("@")
            target_user = await resolve_username(
                context, username, update.effective_chat.id
            )

        except Exception as e:
            await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")
            return
    else:
        # Режим без аргументов - используем отправителя команды
        target_user = update.effective_user

    if not target_user:
        await update.message.reply_text("❌ Пользователь не найден!")
        return

    # Выполняем откат действий
    success = await undo_recent_actions(
        update, context, target_user.id, update.effective_chat.id
    )

    if success:
        await update.message.reply_text(
            f"✅ Действия для пользователя {target_user.mention_markdown_v2()} успешно отменены!",
            
        )
    else:
        await update.message.reply_text("❌ Не найдено действий для отмены")


async def is_user_admin(update: Update, context: CallbackContext) -> bool:
    """Проверяет, является ли пользователь администратором чата"""
    user = update.effective_user
    chat = update.effective_chat

    # Для личных сообщений всегда разрешаем
    if chat.type == "private":
        return True

    admins = await context.bot.get_chat_administrators(chat.id)
    return any(admin.user.id == user.id for admin in admins)


async def resolve_username(
    context: ContextTypes.DEFAULT_TYPE, 
    username: str, 
    chat_id: int
) -> User:
    """Преобразует username в объект User"""
    # Нормализация username
    username = username.lstrip('@')
    
    # Проверка кэша
    cache = context.bot_data.setdefault("username_cache", {})
    if username in cache:
        return cache[username]
    user_id = db.execute("SELECT user_id FROM cache WHERE user_name = ?", (username,)).fetchone()[0]
    print(user_id)
    try:
        
        chat_member = await context.bot.get_chat_member(user_id=user_id, chat_id=chat_id)
        user = chat_member.user
        cache[username] = user
        return user
    except Exception:
        pass
    
    # Поиск по участникам чата (для групп)
    try:
        async for member in context.bot.get_chat_members(chat_id):
            if member.user.username and member.user.username.lower() == username:
                cache[username] = member.user
                return member.user
    except Exception:
        pass
    
    raise ValueError(f"Пользователь @{username} не найден в чате")

async def get_last_actions(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    
    target_user = await resolve_username(
                context, context.args[0], update.effective_chat.id
            )
    cache_records = db.execute(
        "SELECT user_id, chat_id, user_name, action, message, message_label, timestamp FROM cache WHERE user_id=? AND chat_id=?",
        (target_user.id, update.effective_chat.id),
    ).fetchall()
    if not cache_records:
        return

    for record in cache_records:
        _, _, user_name, action, text, _label, date = record
        mention = "from [" + user_name + "](tg://user?id=" + str(target_user.id) + ")\n"
        if text:
            pass
            # await register_new_datachunk({"neutral": text})
        if action.count("ban") > 0 and not await _validate_admin(update):
            # await context.bot.unban_chat_member(chat_id=int(chat_id), user_id=user_id)
            # await update.message.reply_text(text=mention + text, parse_mode="Markdown")
            # return
            pass
        if action.count("mute") > 0 and not await _validate_admin(update):
            # await context.bot.restrict_chat_member(
            #     chat_id=update.message.chat.id,
            #     user_id=int(user_id),
            #     permissions=unmute_permission,
            # )
            pass
        if action.count("del") > 0 and text:
            await update.message.reply_text(text=mention + text, parse_mode="Markdown")
    return


async def undo_recent_actions(
    update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: int
) -> bool:
    """Откатывает последние действия для пользователя (заглушка)"""
    # Ваша реализация логики отката действий
    cache_records = db.execute(
        "SELECT user_id, chat_id, user_name, action, message, message_label, timestamp FROM cache WHERE user_id=? AND chat_id=?",
        (user_id, chat_id),
    ).fetchall()
    if not cache_records:
        await update.effective_chat.send_message("Nothing to restore")
        return
    temp = db.execute(
        "DELETE FROM cache WHERE user_id=? AND chat_id=?",
        (user_id, chat_id),
    )
    db.conn.commit()
    user_name = update.message.from_user["username"]
    for record in cache_records:
        _, _, _, action, text, _label, date = record
        mention = "from [" + user_name + "](tg://user?id=" + str(user_id) + ")\n"
        if text:
            await register_new_datachunk({"neutral": text})
        if action.count("ban") > 0 :
            await context.bot.send_message(text=f'unban in {int(chat_id)} user {user_id}', chat_id=chat_id)
            await context.bot.unban_chat_member(chat_id=int(chat_id), user_id=user_id, only_if_banned=True)
        if action.count("mute") > 0 :
            await context.bot.restrict_chat_member(
                chat_id=update.message.chat.id,
                user_id=int(user_id),
                permissions=unmute_permission,
            )
        if action.count("del") > 0 and text:
            await context.bot.send_message(
                chat_id=chat_id, text=mention + text, parse_mode="Markdown"
            )

    # Возвращает True если действия были отменены, False если нечего отменять
    print(f"Откат действий для user_id={user_id} в чате {chat_id}")
    return True


def main() -> None:
    application = (
        Application.builder().token(config.BOT_TOKEN).build()  # Токен вынесен в конфиг
    )
    # bot = application.context.bot.get_bot()
    # Регистрация обработчиков
    handlers = [
        CommandHandler("start", start),
        CommandHandler("getlastmessages", get_last_actions), 
        CommandHandler("help", help_command),
        CommandHandler("appellate", appellate),
        CommandHandler("rules", set_rules),
        CommandHandler("insult", insult),
        CommandHandler("ad", ad),
        CommandHandler("porn", porn),
        CommandHandler("politics", politics),
        CommandHandler("setmuteduration", set_mute_duration),
        CallbackQueryHandler(button_click),
        InlineQueryHandler(inline_query),
        MessageHandler(filters.TEXT, echo),
    ]
    application.add_handlers(handlers)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
