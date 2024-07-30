def rangeoflist(nums):
    if len(nums) < 3:
        return "Range Determination not Possible"
    max = 0
    min = 0

    for num in nums:
        if num > max:
            max = num
        if num < min:
            min = num
    
    return max - min

if __name__ == "__main__":
    print(rangeoflist([5, 3, 8, 1, 0, 4]))