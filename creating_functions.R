# Introduction ------------

  # functions are an essential part of programming
  # they let you re-use functionality without re-writing code
  # not only are they efficient, they make your code easier to read

# First Example - addition ------------

  # In R, a simple function to add two numbers would be written like this
  # first we declare the name of our function and add the assignment operator
  # we then state that we are creating a function
  # then declare its arguments inside parentheses
  # the code itself is entered between curly brackets
  addition <- 
    function (a,b){
      a + b
    }
  
  # then it is executed as below
  addition(1,4)
  
  # if you change the arguments, the output will alter accordingly
  addition(3,10)
  
# Second Example - retrieval from vector ------------

  # let's look at a slightly more complex example using vectors
  pets <- c("Dog","Cat","Lion","Mouse")
  numbers <-  c(2,3,4,5,6,7,8)
  
  # this function will return the third item of a given vector
  # R indexes (counts from) 1 not 0
  # this is common in maths/stats focused languages but uncommon generally
  item3 <- 
    function (vector){
      vector[3]
    }
  
  item3(pets)
  item3(numbers)
  
  # if we wanted to be able to choose which item was returned, we can do that too!
  itemX <- 
    function (vector, index){
      vector[index]
    }
  
  itemX(pets,2)
  itemX(pets,3)

# Third Example - using packages in functions ------------

  # let's do something more R-specific and use a package to interact with the pets vector
  # stringer (https://stringr.tidyverse.org/) is part of the tidyverse (https://www.tidyverse.org/)
  # if you are unfamiliar with R packages these will help: https://www.dataquest.io/blog/install-package-r/
  library(stringr)
  
  # this function will search a given vector for a provided string
  # the output will show which index the match occurred at plus what characters it starts and ends at
  present <- 
  function (vector,search) {
    str_locate(vector, search)
  }
  
  present(pets,"Lion")
  
# Fourth Example - handling errors ------------
  
  # Looking at our addition function from earlier, it handles numbers well
  # if we use string values instead however, we will have an error
  # this is because + is purely an arithmetic operator in R, so it expect numbers
  addition("Tea","Time")
  
  # to mitigate unwanted values and/or errors, we can include more code as below
  # this is called defensive programming
  addition <- 
    function (a,b){
      if(is.character(a) | is.character(b)){ #this checks if a or b are characters
        paste0(a," ",b) # paste and paste0 concatenate values together
      } else {
        a + b # if there are no characters, our function executes the original code
      }    
    }
  
  # calling the function now doesn't have any errors if there is a string
  # numbers will return the sum as originally
  addition(1,4)
  
  # while strings or a mix of strings and numbers will concatenate with a space between
  addition("Tea","Time")
  addition(10,"Time")
  