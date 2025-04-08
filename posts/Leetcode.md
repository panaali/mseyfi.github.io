# Array and String

## Two pointers

> [!note]
> Start the pointers at the edges of the input. Move them towards each other until they meet.

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

> [!note]
> Move along both inputs simultaneously until all elements have been checked.

```python
function fn(arr1, arr2):
    i = j = 0
    while i < arr1.length AND j < arr2.length:
        Do some logic here depending on the problem
        Do some more logic here to decide on one of the following:
            1. i++
            2. j++
            3. Both i++ and j++

    // Step 4: make sure both iterables are exhausted
    // Note that only one of these loops would run
    while i < arr1.length:
        Do some logic here depending on the problem
        i++

    while j < arr2.length:
        Do some logic here depending on the problem
        j++
```

Similar to the first method we looked at, this method will have a linear time complexity of $O(n+m)$ if the work inside the while loop is 
$O(1)$, where $n = arr1.length$ and $m = arr2.length$. This is because at every iteration, we move at least one pointer forward, and the pointers cannot be moved forward more than $n + m$ times without the arrays being exhausted. Let's look at some examples.
