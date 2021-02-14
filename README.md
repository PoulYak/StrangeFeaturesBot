![DLS](https://static.wixstatic.com/media/66c28f_824e4300a35b4670bd345218eedc211a~mv2.jpg/v1/fit/w_2500,h_1330,al_c/66c28f_824e4300a35b4670bd345218eedc211a~mv2.jpg)


# **Bot with CycleGan**

### _Цель_
Разобрать и изучить архитектуру CycleGan

### _Задачи_:
- Изучить структуру CycleGan
- Решить существующую задачу
- Придумать свою задачу и осуществить её решение
- Сделать интерфейс для визуализации





### _Сбор данных_
Все данные я брал из отрытых источников - kaggle, google
В дальнейшем приходилось отбирать данные подходящие для моей задачи

### _Повтор существующей задачи_ 
В качестве уже решенной задачи я выбрал преобразования изображения со спутника в карты

![](images/img_2.png)

![](images/img_3.png)

![](images/img_4.png)

![](images/img_5.png)


### _Моя задача_
Для своей задачи я решил преобразовывать изображения в мультяшный стиль.
Сначала решил взять стиль мультфильма **Рик и Морти**, но из этого ничего не вышло

![](images/img_6.png)  
_(самое адекватное, что получилось)_

Теперь я решил искать что-то попроще и нашел более подходящую подборку картинок с мультяшным стилем,
он был рассчитан только на девушек, поэтому получается интересно.

![](images/img_7.png)  ![](images/img_8.png)  


Вдобавок я пытался сделать стиль рисунка карандашом, но всё получалось страшно,
поэтому передумал, но в боте оставил



### _Интерфейс_ 
В качестве интерфейса своей программы я решил использовать телеграм бота - удобно, быстро, легко

![Bot](images/img.png)


### _Создание бота и размещение_
Для создания бота я использовал библиотеку pyTelegramBotApi 

![AWS](images/img_1.png) Разместил его на virtual machine AWS EC2


# [**Бот**](https://t.me/StrangeFeaturesBot) 
_@StrangeFeaturesBot_

 
