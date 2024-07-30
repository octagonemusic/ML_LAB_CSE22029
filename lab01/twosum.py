def twosum(nums, target):
    count = 0

    for num1 in nums:
        for num2 in nums:
            if num1 != num2:
                sum = num1+num2
                if sum == target:
                    count = count + 1

    return int(count/2)

if __name__ == "__main__":
    print(twosum([2, 7, 4, 8, 3, 6], 10))