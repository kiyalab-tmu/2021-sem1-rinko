#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 片方動いて、片方が止まる的なプログラムは難しい！同じペースで動作するようにしたほうがプログラムしやすい




"""
軸要素の選択
順に見て、最初に見つかった異なる2つの要素のうち、
大きいほうの番号を返す
全部同じ要素の場合は -1 を返します。
"""
def pivot(arr, i, j):
    k = i + 1
    while k<=j and arr[i]==arr[k]: 
        k = k + 1
    if k>j:
        return -1
    if arr[i]>=arr[k]:
        return i
    return k


"""
パーティション分割
a[i]～a[j]の間で、x を軸として分割します。
x より小さい要素は前に、大きい要素はうしろに来ます。
大きい要素の開始番号を返します。
"""
def partition(arr, i, j, x):
    l = i
    r = j

    #検索が交差するまで繰り返します
    while l<=r:
        # 軸要素以上のデータを探します
        while l<=j and arr[l]<x:
            l = l + 1

      # 軸要素未満のデータを探します
        while r>=i and arr[r]>=x:
            r = r - 1

        if l>r:
            break
        t = arr[l]
        arr[l] = arr[r]
        arr[r] = t
        l = l + 1
        r = r + 1
    return l





def quicksort(arr, i, j):
    if i == j:
        return arr
    p = pivot(arr, i, j)
    if p != -1:
        k = partition(arr, i, j, arr[p])
        quicksort(arr, i, k-1)
        quicksort(arr, k, j)
    return arr





input_array = list(map(int, input().split(' '))) # 入力 ' 1 2 3 4 5' を想定
print(input_array)
print(quicksort(input_array, 0, len(input_array)-1))
