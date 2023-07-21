/********************************************/
/********* A simple search algorithm ********/
/********************************************/  

// imagine this is a website and someone wants to search our data
// in this case, to see if details on a certain programming language is available
// The list below represents our data
dataStore = ["java", "html", "javascript", "css", 
            "c#", "c++", "c", "rust", "julia", "r"];

// and here's our user searching on the site
userSearch = prompt("What programming language do you want details of? ").toLowerCase();

/*****************************/
/***** Iterative Example *****/
/*****************************/ 

// with iteration, this works the way you would expect - a for loop
// to make sure we could use it with any data, it would be in a function
function searchIt(array, term) {
  for (let i = 0; i < array.length; i++) {
    if (array[i] == term){
      return console.log(i)
    } else if (i == array.length){
      return console.log(-1)
      }
  }  
}     

// searching the array
searchIt(dataStore, userSearch);

/*****************************/
/***** Recursive Example *****/
/*****************************/  

// recursively, this function works by gradually reducing the size of the problem
// each time it searches it is checking a smaller subsection of the data

// first we need a function to let them check the data
function searchRec(array, position, total, term) {
  // then checks that it hasn't searched whole array already
  // this is the base case - the thing that prevents a stack overflow
  if (total < 1) {
    return console.log(-1)
  // then checks if current element equals the search term
   // if it does, returns index (which is = to position)
  } else if (array[position] == term) {
    return console.log(position)
  // if neither are true, searches again with different position
  // takes 1 from total and adds 1 to position
  // This lets it check through array elements looking for the term
  } else {
    return searchRec(array, position + 1, total - 1, term)
  }
}

// searching the array
searchRec(dataStore, 0, dataStore.length, userSearch);