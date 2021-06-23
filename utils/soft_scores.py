from collections import Counter

target = [1, 1, 2, 2, 3, 3, 5, 5, 5, 5]
#target = Counter(target)
result = []
for t in list(set(target)):
    score = 0.0
    for i in range(len(target)):
        temp = 0
        for j in range(len(target)):
            if i == j:
                continue
            if t == target[j]:
                temp += 1
        score += min(temp / 3.0, 1.0)
    result.append(score / len(target))

print(result)
print(list(set(target)))
