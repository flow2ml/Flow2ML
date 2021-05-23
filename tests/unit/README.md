# Unit Tests


## Unit Tests
<p>Unit testing refers to a method of testing where software is broken down into different components (units) and each unit is tested functionally and in isolation from the other units or modules.</p>

<p>A unit here refers to the smallest part of a system that achieves a single function and is testable. The goal of unit testing is to verify that each component of a system performs as expected which in turn confirms that the entire system meets and delivers the functional requirements.</p>
 

## Coverage
<p>Coverage.py is a tool for measuring code coverage of Python programs. It monitors your program, noting which parts of the code have been executed, then analyzes the source to identify code that could have been executed but was not.</p>

<p>Coverage measurement is typically used to gauge the effectiveness of tests. It can show which parts of your code are being exercised by tests, and which are not.</p>


## Running unit tests
<p>Run the test suite using coverage to run the unittest runner under coverage. Run the unit tests by executing the following command:</p>

```
coverage run -m unittest discover -s .\tests\unit\
```


## Code coverage
<p>View the coverage report in the terminal by executing:</p>

```
coverage report
```
<p>For a nicer presentation, use the following command to get annotated HTML listings detailing missed lines:</p>

```
coverage html
```