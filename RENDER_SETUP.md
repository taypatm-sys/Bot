# Запуск бота на Render

## Почему текущий запуск завершился ошибкой

Render смог установить зависимости и запустить `bot.py`, но в загруженной
версии репозитория не нашел файл `app/handlers.py`.

В корне GitHub-репозитория должны находиться:

```text
bot.py
requirements.txt
render.yaml
app/
  __init__.py
  config.py
  copywriter.py
  formatting.py
  handlers.py
  models.py
  publisher.py
  scheduling.py
  storage.py
```

Не должно быть лишней внешней папки. При открытии репозитория на GitHub файл
`bot.py` и папка `app` должны быть видны сразу.

## Правильная настройка бесплатного сервиса

Проект подготовлен для бесплатного `Web Service`. Встроенная страница
`/health` открывает обязательный порт Render, а Telegram-бот одновременно
работает через polling.

1. Загрузите в GitHub все файлы из этой папки.
2. Убедитесь, что на GitHub открывается `app/handlers.py`.
3. В Render выберите `New`, затем `Blueprint`.
4. Подключите нужный GitHub-репозиторий.
5. Render прочитает файл `render.yaml` и предложит создать бесплатный Web Service.
6. Введите значения пяти закрытых переменных:

```text
TELEGRAM_BOT_TOKEN
GEMINI_API_KEY
ADMIN_TELEGRAM_ID
CHANNEL_ID
CONTACT_USERNAME
```

7. Подтвердите создание сервиса.

`CHANNEL_ID` должен содержать username канала, например `@taypa_print`, а не
username самого бота.

## Как понять, что все работает

Успешный журнал содержит примерно такие строки:

```text
Render deploy check passed.
Start polling
Run polling for bot
```

В журнале также появится строка `Health server is listening on port 10000`.

## Как не дать бесплатному сервису уснуть

Render усыпляет бесплатный Web Service после 15 минут без входящих HTTP-запросов.
Чтобы бот продолжал работать:

1. Дождитесь успешного запуска и скопируйте адрес сервиса вида
   `https://taypa-telegram-post-bot.onrender.com`.
2. Зарегистрируйтесь бесплатно на `uptimerobot.com`.
3. Нажмите `Add New Monitor`.
4. Выберите тип `HTTP(s)`.
5. Вставьте адрес с путем `/health`, например
   `https://taypa-telegram-post-bot.onrender.com/health`.
6. Выберите интервал 5 минут и сохраните монитор.

## Ограничения бесплатной версии

Бесплатная файловая система Render временная. После нового деплоя, перезапуска
или полного засыпания может исчезнуть локальная очередь SQLite и измененный
через Telegram шаблон. Основной шаблон из GitHub будет создан заново. Не
планируйте критичные публикации на долгое время без проверки команды `/queue`.

Не добавляйте файл `.env` в GitHub. Токен Telegram и ключ Gemini вводятся только
в разделе Environment на Render.
