def chbevl(x, array, n):
    
    b0 = array[0]
    b1 = 0
    i = 1
    j = n - 1
    
    while True:
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + array[i]
        i += 1
        j -= 1

        if j == 0:
            break
                
    return 0.5 * (b0 - b2)