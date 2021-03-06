# Хакатон по искусственному интеллекту Самара
## Кейс №1: Защита редких животных

На основании представленных дата-сетов создана модель машинного обучения для распознавания амурских тигров (Сихотэ-Алинский природный заповедник) и дальневосточного леопарда (национальный парк «Земля леопарда»)

Использование фотоловушек для автоматической съемки видов животных в дикой природе - один из самых эффективных инструментов мероприятий по сохранению биоразнообразия. Эти устройства позволяют проводить точный мониторинг больших участков природных территорий в беспрецедентных масштабах. Однако фотоловушки генерируют слишком большой объем данных, что сильно затрудняет их анализ.

Благодаря последним разработкам в области машинного обучения и компьютерного зрения возможно получить инструменты для распознавания не только видов животных, но и отдельных особей. Появление подобных алгоритмов поможет решению более глобальных задач распознавания всех отдельных особей особо охраняемых видов животных.

## Для установки зависимостей выполнить:

    python3 -m pip install -r requirements.txt

## Для загрузки детектора выполнить:

    wget https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb

## Для загрузки весов модели выполнить:

    wget https://www.dropbox.com/s/sre6ybq2l1glzet/weights.pth


## Для запуска сервера выполнить:
    
    python3 web.py

## Для запуска классификатора выполнить:

    python3 main.py

## Если не запускается так то можно запустить через Google Collab по этой ссылке:
            
    https://colab.research.google.com/drive/1N9QenRTbu26TvQFeipULjAOUCe2uVJnR?usp=sharing
  
 Нужно будет загрузить папку с фотографиями в Google Collab выполнить все кроме последней ячейки, в последней ячейке задать путь к папке с фотографиями, запустить последнюю я чейку в результате создастся файл labels.csv 
