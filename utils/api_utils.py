def extract_args_kwargs(command,not_passed_token="not_passed"):
    args = [arg.split('=')[0] for arg in command if len(arg.split('=')) == 1]
    args = args[1:]
    kwargs = dict()
    for arg in command:
        key_value = arg.split('=')
        if len(key_value) == 2 and key_value[1] != not_passed_token:
            kwargs |= {key_value[0]:key_value[1]}
    return args,kwargs