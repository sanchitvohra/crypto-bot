def linear_decay(initial_value, decay_value, min_value):
    decayed_values = [initial_value]
    while(1):
        next_value = decayed_values[-1] - decay_value
        if next_value <= min_value:
            decayed_values.append(min_value)
            return decayed_values
        else:
            decayed_values.append(next_value)

def step_linear_decay(initial_value, decay_values, min_value, thresholds):
    decayed_values = [initial_value]
    for decay_value, threshold in zip(decay_values, thresholds):
        for i in range(threshold):
            next_value = decayed_values[-1] - decay_value
            if next_value <= min_value:
                decayed_values.append(min_value)
                return decayed_values
            else:
                decayed_values.append(next_value)
    remaining_values = linear_decay(decayed_values[-1], decay_values[-1], min_value)
    return decay_values.append(remaining_values)
    

    