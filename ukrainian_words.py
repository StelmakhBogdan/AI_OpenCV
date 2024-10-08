import random

animals = [
      "Лев",
      "Ведмідь",
      "Слон",
      "Тигр",
      "Зебра",
      "Вовк",
      "Лосось",
      "Кінь",
      "Бик",
      "Крокодил",
      "Жираф",
      "Мавпа",
      "Либідь",
      "Сова",
      "Лелека",
      "Їжак",
      "Верблюд",
      "Корова",
      "Кіт",
      "Пес",
      "Леопард",
      "Жук",
      "Кролик",
      "Олень",
      "Косуля",
      "Ласка",
      "Кенгуру",
      "Коник",
      "Панда",
      "Орел",
      "Лебідь",
      "Фламінго",
      "Страус",
      "Кобра",
      "Ящірка",
      "Шимпанзе",
      "Гієна",
      "Фламінго",
      "Ігуана",
      "Бобер",
      "Єнот",
      "Слон",
      "Сокіл",
      "Мишка",
      "Рись",
      "Пінгвін",
      "Лама",
      "Собака",
      "Півень",
      "Кріль",
      "Панда",
      "Жираф",
      "Ласка",
      "Олень",
      "Кролик",
      "Лисиця",
      "Косуля",
      "Сова",
      "Їжак",
      "Верблюд",
      "Лис",
      "Гепард",
      "Рись",
      "Гієна",
      "Лебідь",
      "Олень",
      "Косатка",
      "Тюлень",
      "Білка",
]

officers_characteristics = [
    'веселий', 'сумний', 'надійний', 'брехливий', 'скупий', 'скучний', 'тліючий', 'жвавий', 'замкнений', 'енергійний',
    'лінивий', 'сміливий', 'боязкий', 'спокійний', 'невпевнений', 'комунікабельний', 'самодостатній', 'дружелюбний',
    'агресивний', 'гострозір',
    'оптимістичний', 'песимістичний', 'конфліктний', 'гармонійний', 'амбітний', 'скромний', 'самовпевнений',
    'неорганізований', 'організований',
    'креативний', 'раціональний', 'імпульсивний', 'мудрий', 'нерішучий', 'емоційний', 'логічний', 'інтуїтивний',
    'рішучий', 'шанобливий',
    'недотепний', 'зігрішливий', 'справжній', 'підлеглий', 'лідер', 'невпевнений', 'працьовитий', 'ледачий',
    'врівноважений', 'незалежний',
    'пристрасний', 'задумливий', 'завзятий', 'байдужий', 'альтруїстичний', 'егоїстичний', 'практичний', 'фанатичний',
    'толерантний',
    'нетерплячий', 'спокійний', 'зворушливий', 'безвідповідальний', 'вірний', 'непостійний', 'надійний', 'ненадійний',
    'цілеспрямований',
    'розсіяний', 'помітний', 'непомітний', 'справедливий', 'несправедливий', 'спонтанний', 'стриманий', 'нудний',
    'цікавий', 'безпечний',
    'небезпечний', 'сором\'язливий', 'несором\'язливий', 'лагідний', 'агресивний', 'романтичний', 'практичний',
    'винахідливий', 'стабільний',
    'лабільний', 'піддатливий', 'непіддатливий', 'володій', 'підлеглий', 'дотепний', 'недотепний', 'зворотний',
    'ініціативний', 'безініціативний',
    'естетичний', 'жорстокий', 'ніжний', 'тверезий', 'п\'яний', 'витончений', 'грубий', 'пристрастий', 'злодій',
    'справедливий'
]

professions_list = [
    'актор', 'архітектор', 'бібліотекар', 'бухгалтер', 'ветеринар', 'геолог', 'громадський діяч', 'дизайнер',
    'дослідник', 'журналіст',
    'інженер', 'консультант', 'лікар', 'музикант', 'педагог', 'програміст', 'психолог', 'секретар', 'співак', 'таксист',
    'фермер', 'фотограф', 'юрист', 'бармен', 'вчитель', 'гравець', 'декоратор', 'директор', 'зоолог', 'економіст',
    'касир', 'кухар', 'лісник', 'майстер', 'модель', 'оператор', 'парикмахер', 'поліцейський', 'режисер', 'суддя',
    'технік', 'тренер', 'художник', 'цілитель', 'шахтар', 'швея', 'військовий', 'активіст', 'брокер', 'візажист',
    'гід', 'гітарист', 'драматург', 'декан', 'ентомолог', 'живописець', 'кондитер', 'культорганізатор', 'маркетолог',
    'офіціант',
    'пожежник', 'проректор', 'публіцист', 'соліст', 'фармацевт', 'футболіст', 'штурман', 'імунолог', 'булочник',
    'вітеринар',
    'географ', 'діловод', 'жокей', 'каменяр', 'кіноактор', 'модельєр', 'оператор', 'пекар', 'пілот', 'програміст',
    'режисер', 'скульптор', 'співак', 'стоматолог', 'таксист', 'транспортер', 'фармацевт', 'фермер', 'хореограф',
    'шахтар',
    'інструктор', 'блогер', 'психотерапевт', 'маркетолог', 'геолог', 'економіст', 'програміст', 'технолог', 'інженер',
    'модель',
    'медсестра', 'будівельник', 'вчитель', 'археолог', 'відвідувач', 'кореспондент', 'лікар', 'продюсер', 'ресторатор',
    'стоматолог',
    'садівник', 'шафер', 'актор', 'бізнесмен', 'велосипедист', 'власник', 'кондуктор', 'провізор', 'співробітник',
    'танцюрист', 'футболіст'
]

# Create dict for conversation
keyboard_layout = {
    'а': 'f', 'б': ',', 'в': 'd', 'г': 'u', 'ґ': '`', 'д': 'l', 'е': 't', 'є': '\'',
    'ж': ';', 'з': 'p', 'и': 'b', 'і': 's', 'ї': ']', 'й': 'q', 'к': 'r', 'л': 'k',
    'м': 'v', 'н': 'y', 'о': 'j', 'п': 'g', 'р': 'h', 'с': 'c', 'т': 'n', 'у': 'e',
    'ф': 'a', 'х': '[', 'ц': 'w', 'ч': 'x', 'ш': 'i', 'щ': 'o', 'ь': 'm', 'ю': '.',
    'я': 'z', 'А': 'F', 'Б': '<', 'В': 'D', 'Г': 'U', 'Ґ': '~', 'Д': 'L', 'Е': 'T',
    'Є': '\"', 'Ж': ':', 'З': 'P', 'И': 'B', 'І': 'S', 'Ї': '}', 'Й': 'Q', 'К': 'R',
    'Л': 'K', 'М': 'V', 'Н': 'Y', 'О': 'J', 'П': 'G', 'Р': 'H', 'С': 'C', 'Т': 'N',
    'У': 'E', 'Ф': 'A', 'Х': '{', 'Ц': 'W', 'Ч': 'X', 'Ш': 'I', 'Щ': 'O', 'Ь': 'M',
    'Ю': '>', 'Я': 'Z'
}

# Generating 50 lines for file, u can change range for generating more words
lines = []
single_words_lines = []
for _ in range(75):
    rank = random.choice(animals).capitalize()
    characteristic = random.choice(officers_characteristics)
    profession = random.choice(professions_list)

    # Convert every symbol from keyboard
    rank_converted = ''.join(keyboard_layout.get(letter, letter) for letter in rank)
    characteristic_converted = ''.join(keyboard_layout.get(letter, letter) for letter in characteristic)
    profession_converted = ''.join(keyboard_layout.get(letter, letter) for letter in profession)

    lines.append(
        f'{rank} {characteristic} {profession} - {rank_converted} {characteristic_converted} {profession_converted}')
    single_words_lines.append(
        f'{rank} {characteristic} {profession}')

with open("passwords.txt", "w", encoding="utf-8") as file:
    for line in lines:
        file.write(line + '\n')

with open("passwords_words.txt", "w", encoding="utf-8") as word_file:
    for single_word in single_words_lines:
        word_file.write(single_word + '\n')
