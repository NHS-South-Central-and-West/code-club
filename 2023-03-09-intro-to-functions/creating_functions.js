/* If you want to run JavaScript code and can't locally on your machine,
go to www.blank.org, right click then press 'Inspect'. That will open the 
Developer Tools pane - you want to be on the tab that says 'Console'. */

/**************************************************/
/***************** INTRODUCTION *******************/
/**************************************************/

// functions are an essential part of programming
// they let you re-use functionality without re-writing code
// not only are they efficient, they make your code easier to read

/**************************************************/
/******************* EXAMPLE 1 ********************/
/**************************************************/

// in JavaScript, a simple function to add two numbers would be written like this
// first we declare that we are creating a function
// we then provide the name of our function and declare its arguments inside parentheses
// the code itself is entered between curly brackets
// return ends the execution of our function and specifies the returned value
function addition(a,b){
    return a + b;
}

// then it is executed as below
addition(1,4);

// if you change the arguments, the output will alter accordingly
addition(10,40);

/**************************************************/
/******************* EXAMPLE 2 ********************/
/**************************************************/

// let's look at a slightly more complex example using arrays
let pets = ["Dog","Cat","Lion","Mouse"]
let numbers = [2,3,4,5,6,7,8]

// this function will return the third item of an array
// like most programming languages, JavaScript indexes (counts from) 0
function item3(array){
    return array[2];
}

item3(pets);
item3(numbers);

// if we wanted to be able to choose which item was returned, we can do that too!
function itemX(array, index){
    return array[index];
  }

itemX(pets,2);
itemX(pets,3);

/**************************************************/
/******************* EXAMPLE 3 ********************/
/**************************************************/

// let's do something more JavaScript-specific and add a new item to our pets array
function addItem(array){
    let add = prompt("Please enter the new pet"); // prompt asks a user to input a value
    array.push(add); // push adds the value to our array
    console.log(pets); // this outputs the new array to the console
}

// now the function will ask you for input and add that input to the array
// console.log will also output the result to your console in JavaScript
addItem(pets);

/**************************************************/
/******************* EXAMPLE 4 ********************/
/**************************************************/

// Looking at our addition function from earlier, it handles numbers well
// if we use string values instead however, we will get a contatenation instead
// this is because JavaScript treats + as an arithmetic operator with numbers, but changes behaviour contextually
addition("Tea","Time");

// to mitigate unwanted values and/or errors, we can include more code as below - this is called defensive programming
// this simple example uses an if/else statement to control what the output will be conditionally
function addition(a,b){
    if(isNaN(a) | isNaN(b)){ 
        return String(a) + " " + String(b);
    } else {
        return a + b;
    }    
}

// numbers will still return the sum
addition(1,4);

// while strings or a mix of strings and numbers will concatenate with a space between
addition("Tea","Time");

addition(10,"Time");