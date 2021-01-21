## RDS Diploma: Heart Failure Prediction.

Построение и быстрое развертывание модели предсказания выживаемости пациентов с болезнью сердца с помощью МО.

Развернутый прототип модели МО:

https://share.streamlit.io/testdriver87/sf_diploma/main/app.py



![](https://img.webmd.com/dtmcms/live/webmd/consumer_assets/site_images/article_thumbnails/slideshows/did_you_know_this_could_lead_to_heart_disease_slideshow/650x350_did_you_know_this_could_lead_to_heart_disease_slideshow.jpg)

### Данные для создания модели

За основу я взял публичный [датасет Heart Failure Prediction с Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). Он содержит 299 записей пациентов и 12 числовых и категориальных признака, которые отражают как образ жизни и сопутствующие болезни наблюдаемого пациента, так и параметры анализа крови. 

В данном датасете предлагается предсказать выживаемость пациента с заданным набором признаков.

### Выбор метрики

Проанализировав несколько статей по схожим проектам МО (например https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7), я пришел к выводу, что правильнее всего будет использовать метрику MCC (Matthews correlation coefficient) как основную. Так как она является более надежным статистическим показателем, которая выдает высокое значение только в том случае, если прогноз получил хорошие результаты во всех четырех категориях матрицы путаницы (истинные положительные, ложные отрицательные, истинные отрицательные и ложные положительные), ***пропорционально как размеру положительных элементов, так и размеру отрицательных элементов в наборе данных***. Метрику AUC будем использовать как дополнительное подтверждение.

### Описание маршрута выполнения проекта

Для проведения EDA я использовал пакеты [SweetViz](https://github.com/fbdesignpro/sweetviz) и [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling), полностью автоматизирующие и значительно облегчающие, работу с сырыми данными. С автоматически сгенерированными html-отчетами  можно ознакомиться по ссылкам ниже:

https://github.com/testdriver87/sf_diploma/blob/main/PANDAS_PROFILING_REPORT.html

https://github.com/testdriver87/sf_diploma/blob/main/PANDAS_PROFILING_REPORT.html

Для предварительной оценки важности признаков я использовал пакет [LOFO](https://github.com/aerdem4/lofo-importance). Он  вычисляет важность набора признаков на основе выбранной метрики для модели выбора, итеративно удаляя каждый признак из набора и оценивая производительность модели с помощью схемы проверки выбора, основанной на выбранной метрике.

Для создания модели я использовал пакет [PyCaret](https://github.com/pycaret/pycaret). Этот пакет машинного обучения с открытым исходным кодом на Python для обучения и развертывания моделей с учителем и без учителя в low-code среде. По сравнению с другими открытыми библиотеками машинного обучения, PyCaret – это low-code альтернатива, которая поможет заменить сотни строк кода всего парой слов. Pycaret автоматизирует такие рутинные операции как: кроссвалидация, оптимизация гиперпараметров, калибровка модели, подготовка данных (feature engineering, feature selection и т.п.). Все операции, выполняемые PyCaret, последовательно сохраняются в пайплайне полностью готовом для развертывания.

Для развертывания модели я пользовался только средствами Streamlit. С помощью этого модуля я написал головное приложение app.py, а также использовал их же новый сервис https://share.streamlit.io/, который позволяет обойтись без Heroku. Это значительно экономит время при прототипировании.

### Итоги и выводы

...

Основной проблемой этого датасета и других подобных является слишком малое количество записей (пациентов). Если бы на примере хотя бы одной страны или даже региона было бы возможно собрать данные таких анализов в едином виде, то точность модели была бы значительно лучше.

Внимательно прочитав несколько статей по данной тематике я так и не нашел там упоминания случаев успешного применения МО. Причины отсутствия успешных практик применения требуют дальнейшего изучения. 

Что могу предположить и предложить я? 

Например. Несколько недель назад в Твиттере мне попалась реклама следующего мобильного приложения. Оно позволяет загружать, распознавать и расшифровывать медицинские анализы.

<img src="https://i.imgur.com/WhxZcpS.jpg" style="zoom: 25%;" /> <img src="https://i.imgur.com/tEEkCyM.jpg" style="zoom:25%;" />

А что если бы оно могло помимо прочего определять предрасположенность к заболеваниям или к развитию летального исхода для пациента. Но для этого опять же нужно обработать большое количество анализов пациентов. А данные в таком количества имеются только лишь в государственных клиниках. Поэтому скорее можно предложить модуль для приложения системы ЕМИАС в дополнение к электронной медицинской карте.

