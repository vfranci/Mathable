# Mathable
Project to identify gameboard and tiles in a Mathable game using image preprocessing techniques, developed during the Computer Vision course in my 4th year at the Faculty of Mathematics and Computer Science, University of Bucharest. 

## Data structure
The given data consists of four Mathable games. Each game has 50 images, representing each turn of the players. 

## Task 1
The purpose of the first task is identifying the position of the newly placed tile. This is done using a combination of filters to isolate the gameboard and the existing tiles, representing the state of the game at each turn using a matrix and comparing all state matrixes. The accuracy at this stage of deveopement is around 0.9.

## Task 2
The second task of the game is identifying the number on each tile. This is done using template matching, using the templates1 folder. The accuracy at this stage is around 0.6.

