# Array and String

## Two pointers

'''
Start the pointers at the edges of the input. Move them towards each other until they meet.
'''

```python
function fn(arr):
    left = 0
    right = arr.length - 1

    while left < right:
        Do some logic here depending on the problem
        Do some more logic here to decide on one of the following:
            1. left++
            2. right--
            3. Both left++ and right--
```
The strength of this technique is that we will never have more than  $O(n)$ iterations for the while loop because the pointers start $n$ away from each other and move at least one step closer in every iteration. Therefore, if we can keep the work inside each iteration at $O(1)$, this technique will result in a linear runtime, which is usually the best possible runtime. 

'''
Move along both inputs simultaneously until all elements have been checked.
'''
