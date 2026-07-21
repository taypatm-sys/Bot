# Обновление бота на Render

Эта версия работает как бесплатный Render Web Service и хранит очередь,
шаблон и пресеты в постоянной базе Neon PostgreSQL.

## Перед обновлением

Старая версия хранит очередь в локальном SQLite. Эти записи не переносятся в
Neon автоматически. Перед новым деплоем откройте в боте `/queue` и запишите
время важных публикаций или опубликуйте их заранее.

## 1. Обновите файлы в GitHub

Загрузите содержимое папки проекта в корень существующего репозитория с заменой
старых файлов. На главной странице репозитория должны сразу быть видны:

```text
bot.py
requirements.txt
render.yaml
caption_template.txt
app/
```

Внутри `app` должен быть новый файл `template_store.py`.

Не загружайте `.env`, `posts.sqlite3`, токен Telegram, ключ Gemini или строку
подключения Neon.

## 2. Создайте бесплатную базу Neon

1. Откройте `https://console.neon.tech` и зарегистрируйтесь.
2. Нажмите `New Project` и создайте проект, например `taypa-bot`.
3. На странице проекта нажмите `Connect`.
4. Оставьте включенным `Connection pooling`.
5. Скопируйте всю строку, которая начинается с `postgresql://`.

Пример формата:

```text
postgresql://user:password@ep-example-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```

Официальная инструкция Neon:
`https://neon.com/docs/connect/connect-from-any-app`

## 3. Добавьте DATABASE_URL в Render

1. Откройте существующий сервис `bot-y92e` в Render.
2. Слева откройте `Environment`.
3. Нажмите `Add Environment Variable`.
4. В поле Key укажите `DATABASE_URL`.
5. В Value вставьте скопированную строку Neon целиком.
6. Нажмите `Save, rebuild, and deploy`.

Остальные переменные оставьте без изменений:

```text
TELEGRAM_BOT_TOKEN
GEMINI_API_KEY
ADMIN_TELEGRAM_ID
CHANNEL_ID
CONTACT_USERNAME
TIMEZONE
GEMINI_MODEL
BUTTON_TEXT
COPY_LANGUAGE
```

Официальная инструкция Render:
`https://render.com/docs/configure-environment-variables`

## 4. Проверьте команды Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
python bot.py
```

Health Check Path:

```text
/health
```

## 5. Проверьте запуск

Успешный журнал содержит строки:

```text
Render deploy check passed.
Health server is listening on port 10000
Start polling
Run polling for bot
```

После запуска отправьте боту `/check`. Последняя строка должна быть:

```text
База: PostgreSQL
```

Если указано `База: SQLite`, переменная `DATABASE_URL` не сохранена или сервис
не был заново развернут.

## Как не дать бесплатному сервису уснуть

Бесплатный Render Web Service засыпает после 15 минут без входящих запросов.
Настройте внешний HTTP-монитор на адрес:

```text
https://bot-y92e.onrender.com/health
```

Интервал проверки можно поставить 5 минут. Neon сохранит данные даже после
засыпаний, перезапусков и новых деплоев Render.

Ограничения бесплатного Render:
`https://render.com/docs/free`

## Новые возможности

- Постоянная очередь, шаблон и пресеты в Neon.
- Предпросмотр любого запланированного поста.
- Изменение названия, описания, размеров, цены, даты и времени.
- Публикация поста прямо из очереди.
- Создание копии запланированного поста.
- Готовые пресеты и добавление своих через `/presets`.
