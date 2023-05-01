# project
Проект Django для распознавания и классификации картинок овощей и фруктов
Данный проект представляет собой веб-приложение, созданное на базе фреймворка Django, которое использует нейронную сеть для распознавания и классификации картинок овощей и фруктов.

Установка и запуск проекта
Для запуска данного проекта вам необходимо:
1.	Склонировать репозиторий на свой компьютер:
git clone https://github.com/username/repository.git

2.  Перейти в папку с проектом:
cd repository

3.  Установить зависимости:
pip install -r requirements.txt

4.Кроме того, для работы данной нейронной сети необходимо использовать датасет изображений овощей и фруктов. Для этого мы можем воспользоваться набором данных, доступным на Kaggle по ссылке https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition. Для использования данного датасета в проекте, необходимо загрузить его на свой компьютер и поместить в папку dataset внутри проекта Django. После этого, изменять путь к файлу датасета в коде модели нейронной сети не будет требоваться.

5.Запустить миграции:
python manage.py migrate

6.Запустить сервер:
python manage.py runserver

7.Открыть браузер и перейти по адресу http://127.0.0.1:8000/

Использование приложения
После запуска приложения, вы попадете на главную страницу, где сможете загрузить изображение овоща или фрукта для классификации. После загрузки изображения, система выполнит его анализ и выведет результат классификации на экран.
