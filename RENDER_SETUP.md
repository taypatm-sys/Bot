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

Внутри `app` должны быть файлы `template_store.py` и `mockup_generator.py`.

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
GEMINI_IMAGE_MODEL
GEMINI_IMAGE_SIZE
MOCKUP_VARIANTS
REFERENCE_IMPORT_DELAY_SECONDS
REFERENCE_IDLE_INTERVAL_SECONDS
REFERENCE_MAX_ATTEMPTS
REFERENCE_MIN_POOL_SIZE
REFERENCE_ANALYSIS_TIMEOUT_SECONDS
MOCKUP_ANALYSIS_TIMEOUT_SECONDS
BUTTON_TEXT
COPY_LANGUAGE
```

Если новых переменных еще нет, добавьте их со значениями:

```text
GEMINI_IMAGE_MODEL=gemini-3.1-flash-lite-image
GEMINI_IMAGE_SIZE=1K
MOCKUP_VARIANTS=1
REFERENCE_IMPORT_DELAY_SECONDS=5
REFERENCE_IDLE_INTERVAL_SECONDS=300
REFERENCE_MAX_ATTEMPTS=5
REFERENCE_MIN_POOL_SIZE=20
REFERENCE_ANALYSIS_TIMEOUT_SECONDS=90
MOCKUP_ANALYSIS_TIMEOUT_SECONDS=90
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

В PostgreSQL также сохраняется активный черновик на 48 часов. Поэтому кнопки
выбора пресета и ручного ввода продолжат работать после перезапуска Render.

Следующая строка должна показывать параметры нового режима:

```text
Фото на модели: gemini-3.1-flash-lite-image, 1K, 4:5
```

Также отправьте `/references`. Сразу после первого запуска должно быть 63 ссылки
в очереди. Число `Готово` будет постепенно увеличиваться, потому что изображения
загружаются и размечаются по одному в фоне.

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
- Каталог референсов в Neon и загрузка новых TXT-списков через `/references`.
- Создание одной реалистичной фотографии 4:5 из готового макета.
- Бесплатная карточка анализа до запуска платной генерации.
- Дополнительный оригинальный PNG принта с проверкой прозрачности.
- Разные люди, лица, позы и локации в каждой серии.
- Выбор сгенерированной фотографии для нового поста.

## Доступ к генерации изображений

Модель `gemini-3.1-flash-lite-image` не имеет бесплатного API-тарифа. В проекте
Google AI Studio для используемого `GEMINI_API_KEY` необходимо включить биллинг.
Без него весь старый функционал бота продолжит работать, но режим
`Фото на модели` вернет понятное предупреждение.

Стандартная серия создает одно изображение 1K и по текущему тарифу стоит около
0.0336 доллара без учета небольшого входного запроса. Актуальная цена указана на странице:
`https://ai.google.dev/gemini-api/docs/pricing`.
