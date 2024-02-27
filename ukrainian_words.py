import nltk

nltk.download('udhr')
nltk.download('words')

from nltk.corpus import words

ukrainian_words = [word for word in words.words() if
                   all(ord(char) < 128 for char in word) and 'а' <= word <= 'я' and 'і' in word]
ukrainian_words = ukrainian_words[:10000]  # Take the first 10,000 words

with open("ukrainian_words.txt", "w", encoding="utf-8") as file:
    for word in ukrainian_words:
        file.write(word + "\n")


ukrainianWords = [
    'кінь', 'бігає', 'галопом', 'мама', 'тато', 'сонце', 'море', 'вітер', 'день', 'ночі',
    'хмара', 'дерево', 'дім', 'сад', 'трава', 'квітка', 'річка', 'озеро', 'гора', 'поля',
    'ліс', 'птах', 'собака', 'кіт', 'корова', 'конячий', 'кістка', 'корінь', 'зелень', 'фрукт',
    'пташка', 'кролик', 'вовк', 'лисиця', 'вівця', 'коза', 'корівка', 'курка', 'індик', 'курча',
    'молоко', 'сир', 'яйце', 'мед', 'цукор', 'мука', 'сіль', 'олія', 'масло', 'сметана',
    'хліб', 'булка', 'батон', 'пиріг', 'пиріжок', 'печиво', 'торт', 'пончик', 'млинці', 'борщ',
    'суп', 'каша', 'плов', 'пиріг', 'плита', 'піч', 'ковш', 'сковорода', 'каструля', 'тарілка',
    'чашка', 'ложка', 'виделка', 'ніж', 'склянка', 'банка', 'кухня', 'стіл', 'стілець', 'шафа',
    'ліжко', 'диван', 'крісло', 'стіна', 'підлога', 'стеля', 'вікно', 'двері', 'перина', 'ковдра',
    'подушка', 'ковбаса', 'сосиска', 'колбаса', 'бекон', 'сало', 'шинка', 'колбаска', 'паштет', 'м\'ясо'
]