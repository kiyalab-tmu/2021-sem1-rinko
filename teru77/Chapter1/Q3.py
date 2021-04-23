def Missing_Element(arr1,arr2):
    if len(arr1) > len(arr2): 
        Original_array,Missing_array = arr1,arr2   
    elif len(arr1) < len(arr2):
        Original_array,Missing_array = arr2,arr1   
    else:
        print('None')
    
    for ele in Missing_array:
        Original_array.remove(ele)
        
    return Original_array

if __name__ == '__main__':
    sample_input_A,sample_input_B = [2, 3, 4, 5, 6, 7, 5, 8], [6, 8, 7, 4, 5, 2, 3]
    print(Missing_Element(sample_input_A,sample_input_B))