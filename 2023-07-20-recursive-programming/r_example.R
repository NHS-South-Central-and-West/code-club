########## A simple search algorithm #########

# imagine this is a website and someone wants to search our data
# in this case, to see if details on a certain programming language is available
# The list below represents our data
dataStore = c("java", "html", "javascript", "css",
              "c#", "c++", "c", "rust", "julia", "r")

# And here's our user searching on the site

userSearch <- 
  tolower(
    readline(prompt="What programming language do you want details of? ")
  )

###### Iterative Example ######

# With iteration, this works the way you would expect - a for loop
# To make sure we could use it with any data, it would be in a function
searchIt <- function (array, term) {
  for (i in 1:length(array)) {
    if (array[i] == term){
       print(i)
       break
    } else if (i == length(array)) {
       print(-1)
    }
  }  
}   

searchIt(dataStore, userSearch)

###### Recursive Example ######

# recursively, this function works by gradually reducing the size of the problem
# each time it searches it is checking a smaller subsection of the data

# first we need a function to let them check the data
searchRec <- function (array, position, total, term) {
  # then checks that it hasn't searched whole array already
  # this is the base case - the thing that prevents a stack overflow
  if (total < 1) {
    print(-1)
# then checks if current element equals the search term
# if it does, returns index (which is = to position)
  } else if (array[position] == term) {
     print(position)
# if neither are true, searches again with different position
# takes 1 from total and adds 1 to position
# This lets it check through array elements looking for the term
  } else {
    searchRec(array, position + 1, total - 1, term)
  }
}

# searching the array
searchRec(dataStore, 1, length(dataStore), userSearch)