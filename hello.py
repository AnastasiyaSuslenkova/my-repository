import random

print("Введите число массивов")
n = int(input())
max_length = 100*n   # Максимальная длина массива. Можно поменять ее на другое натуральное число,
                     # но оно должно быть не меньше n, иначе не получится создать n массивов разной длины.

Mass = []  # Массив с массивами
rand = []
for i in range(n):
    while True:
        rand[i] = random.randint(1, max_length)
        for j in range(i):
            if rand[i] == rand[j]:
                break
        break
    mas = []
    for j in range(rand[i]):
        mas.append(random.random)   # Добавили j-ый элемент в массив. В задании не сказано, какими числами должны быть заполнены массивы,
                                    # поэтому я заполнила числами от 0 до 1.
        if i % 2 == 0:                                  # Можно сортировать массив прямо во время его создания (так будет быстрее),
            for k in range(j):                          # но если нам нужен исходный (неотсортированный) массив, можно разделить эти два этапа;
                if mas[k] > mas[j]:                     # однако, в задании требуется только массив из отсортированных массивов, поэтому этапы
                    for l in range(j, k, -1):           # создания и сортировки можно совместить.
                        mas[l], mas[l-1] = mas[l-1], mas[l]
                    break
        else:
            for k in range(j):
                if mas[k] < mas[j]:
                    for l in range(j, k, -1):
                        mas[l], mas[l - 1] = mas[l - 1], mas[l]
                    break
    Mass.append(mas)  # Добавили отсортированный массив

