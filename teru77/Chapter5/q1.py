import numpy as np
import re

######### word level #########

#辞書を作成
def Create_a_word_vocabulary():
    with open('time_machine.txt','r',encoding='UTF-8') as f:
        data = f.read()
        data = data.lower() #小文字にする
        data = re.sub(re.compile("[!-/:-@[-`{-~]"), '', data)
    #辞書の作成 
    vocabulary = {}
    for word in data.split():
         vocabulary[word] = vocabulary.get(word, 0) + 1 #キーが存在しなかったら1,存在していたら+1する
    
    #頻度でソート 
    list = sorted(vocabulary.items(),key=lambda x:x[1],reverse=True)   
    
    #listになるのでdictに戻す
    vocabulary.clear()
    vocabulary.update(list)
    
    #頻度が薫物から順にインデックスをつけていく
    for i,key in enumerate(vocabulary.keys()):
        vocabulary[key] = i
    return vocabulary

#インデックスに変換
def Convert_word_to_index():
    
    vocabulary = Create_a_word_vocabulary()
    
    f = open('time_machine.txt','r',encoding='UTF-8')
    lines = f.readlines()
    
    sentences = []
    words = []
    
    for line in lines: 
        if line == '\n':
            continue
        else:
            line = line.lower() #小文字にする
            line = re.sub(re.compile("[!-/:-@[-`{-~]"), '', line)
            for word in line.split():
                words.append(vocabulary[word])
                   
        if len(words)>0:
            sentences.append(words)
        
        words = []
    return sentences

######### character level #########

#辞書を作成
def Create_a_charactor_vocabulary():
    f =open('time_machine.txt','r',encoding='UTF-8')
    lines = f.readlines()
    #辞書の作成 
    vocabulary = {}
    
    for line in lines:
        if line == '\n':
            continue      
        else:
            line = line.lower() #小文字にする
            line = re.sub(re.compile("[!-/:-@[-`{-~]"), '', line)
            line = re.sub(re.compile('[æ…üœç“”‘’]'), '', line) #アルファベット：26種、数字：10種、-(ハイフン)の計37種類のキーを用意
            
            for c in line:
                if c == " " or c == "\n":
                    continue
                else:
                    vocabulary[c] = vocabulary.get(c, 0) + 1 #キーが存在しなかったら1,存在していたら+1する

    #頻度でソート 
    list = sorted(vocabulary.items(),key=lambda x:x[1],reverse=True)  
     
    #listになるのでdictに戻す
    vocabulary.clear()
    vocabulary.update(list)
    
    #頻度が薫物から順にインデックスをつけていく
    for i,key in enumerate(vocabulary.keys()):
        vocabulary[key] = i
    return vocabulary

#インデックスに変換
def  Convert_charactor_to_index():
    
    vocabulary = Create_a_charactor_vocabulary()
    
    f = open('time_machine.txt','r',encoding='UTF-8')
    lines = f.readlines()
    
    sentences = []
    char = []
    
    for line in lines: 
        if line == '\n':
            continue
        else:
            line = line.lower() #小文字にする
            line = re.sub(re.compile("[!-/:-@[-`{-~]"), '', line)
            line = re.sub(re.compile('[æ…üœç“”‘’]'), '', line)
            for c in line:
                if c == " " or c == "\n":
                    continue
                else:
                    char.append(vocabulary[c])  
        if len(char)>0:
            sentences.append(char)
        
        char = []
    return sentences
        
"""
if __name__ == '__main__':
    word_dataset = Convert_word_to_index()
    print(word_dataset[0])
    char_dataset = Convert_charactor_to_index()
    print(char_dataset[0])
"""