import re


def word_tokenizer(lines):
    word_dict = {}   # {"word": num of the word} : no diplicates

    # create new_lines
    new_lines = []   # new_line[n] keep all word which appear in lines[n]
    text_cleaner = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋—￥％…\‘\’]')
    for line in lines:
        words = text_cleaner.sub(' ', line).lower().split()
        new_lines.append(words)
        for word in words:
            if word in word_dict.keys():
                word_dict[word] += 1
            else:
                word_dict[word] = 1

    # soted items(words) according to its frequency
    soted_word_list = sorted(word_dict.items(), key=lambda x:x[1])
    soted_word_list.reverse()

    all_word = []   # keep all word in text according to its frequency
    for i in range(len(soted_word_list)):
        all_word.append(soted_word_list[i][0])

    # replace words with its index in new_lines
    for i in range(len(new_lines)):
        for j in range(len(new_lines[i])):
            new_lines[i][j] = all_word.index(new_lines[i][j])+1
            

    return new_lines

def character_tokenizer(lines):
    char_dict = {}   # {"character": num of the character} : no diplicates
    # create new_lines
    new_lines = []   # new_line[n] keep all character which appear in lines[n]
    text_cleaner = re.compile('[!"#$%&\'\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋—￥％…\‘\’]')
    for line in lines:
        line = text_cleaner.sub('', line).lower()
        chars = []
        for c in line:
            chars
            if c ==' ' or c=='\n':
                continue
            if c in char_dict:
                char_dict[c] += 1
            else:
                char_dict[c] = 1
            chars.append(c)
        new_lines.append(chars)
    
    # soted items(characters) according to its frequency
    soted_char_list = sorted(char_dict.items(), key=lambda x:x[1])
    soted_char_list.reverse()
    print(len(soted_char_list))

    all_char = []   # keep all word in text according to its frequency
    for i in range(len(soted_char_list)):
        all_char.append(soted_char_list[i][0])
    
    # replace words with its index in new_lines
    for i in range(len(new_lines)):
        for j in range(len(new_lines[i])):
            new_lines[i][j] = all_char.index(new_lines[i][j])+1
    
    return new_lines


def main():
    DataPath = 'jimar884/chapter5/TheTimeMachine.txt'
    file = open(DataPath, 'r', encoding='utf-8-sig')   # utf-8-sig removes '/ufeff'
    lines = file.readlines()

    # new_lines = word_tokenizer(lines.copy())
    # for i in range(10):
    #     print(lines[i])
    # print("-------"*10)
    # for i in range(10):
    #     print(new_lines[i])

    new_lines = character_tokenizer(lines)
    for i in range(10):
        print(lines[i])
    print("-------"*10)
    for i in range(10):
        print(new_lines[i])


if __name__=='__main__':
    main()