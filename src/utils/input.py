def get_nonneg_int_input():
    """ Prompt user until they enter nonnegative integer input. """
    while True:
        response = input()  # response is a string
        try:
            val = int(response)
            if val >= 0:
                break
            print('Integer must be non-negative.')
        except:
            print('Please enter an integer.')

    return val
