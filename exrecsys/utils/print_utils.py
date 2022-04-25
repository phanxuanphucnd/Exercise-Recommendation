def print_free_style(message, print_fun=print):
    print_fun("▓  {}".format(message))
    print_fun("")

def print_style_time(message, print_fun=print):
    print_fun("")
    print_fun("⏰  {}".format(message))
    print_fun("")
    
def print_style_warning(message, print_fun=print):
    print_fun("")
    print_fun("⛔️  {}".format(message))
    print_fun("")
    
def print_style_notice(message, print_fun=print):
    print_fun("")
    print_fun("📌  {}".format(message))
    print_fun("")

def print_line(text, print_fun=print):
    print_fun("")
    print_fun("➖➖➖ {} ➖➖➖".format(text.upper()))
    print_fun("")
