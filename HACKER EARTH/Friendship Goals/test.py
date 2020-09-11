for _ in range(int(input())):
    n, k = list(map(int, input().split(" ")))
    sample = [x for x in range(1, n + 1)]

    max = 0

    # print(n,k)

    for i in range(0, len(sample) - 1):
        for j in range(i + 1, len(sample)):
            temp = sample[i] & sample[j]
            if temp < k:
                # print(temp,sample[i],sample[j])
                if max < temp:
                    max = temp

        if max == k - 1:
            break

    print(max)



