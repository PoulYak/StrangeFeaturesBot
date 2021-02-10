import telebot
from telebot import types

from cycle import get_image

with open('token.txt') as fin:
    token = fin.readline()
bot = telebot.TeleBot(token)



@bot.message_handler(commands=['start', 'help'])
def hello(message):
    text = 'Hi) if you want to know what I can just send me a photo'
    bot.send_message(message.chat.id, text)


@bot.message_handler(content_types=['photo'])
def send_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    with open('imag.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)

    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    bn1 = types.KeyboardButton('Pencil style')
    bn2 = types.KeyboardButton('Zebra')
    kb.add(bn1)
    kb.add(bn2)
    bot.send_message(message.chat.id, 'Choose style', reply_markup=kb)
    bot.register_next_step_handler(message, get_photo_style)


def get_photo_style(message):
    get_image('imag.jpg', 'peoples2pencil', str(message.from_user.id))
    doc = open('results/' + str(message.from_user.id) + '/imag.jpg', 'rb')

    bot.send_photo(message.chat.id, doc, reply_markup=types.ReplyKeyboardRemove())


bot.polling(none_stop=True, interval=0)
