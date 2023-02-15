# NASA-Flame-Project
Various files containing the later stages of my work with Rutgers Professor Tse and Jonathan Shi

List of everything fullAnalyze does written above the source code in a comment

If at any point you need to redo an individual folder full of LLL-UV or INTNS images i.e. 20006B2_INTNS_ALL,
then you must empty all related folders (the 20006B2_INTNS_GRAPHS and 20006B2_INTNS_IMAGES)
You can empty the flame only folder if you wish to recalculate the flame images as well

Parameters for Different Image Types [Sigma, Truncate, Center]
- INTNS : 7,3,(720,383)
- LLL-UV : 1,5,(259,268)
- ACME_Scl_ALL : 9,3,(532,758)

How to use:
1) copy paste the code into a python file to use as an import in an execute file
2) In execute file import the fullAnalyze code as [any name]
3) simply put [name].fullAnalyze(mainDir, imgType, Sigma, Truncate, Center)

the mainDir should be an entire folder size such as 20029, 19254
example: mainDir = r'E:\.shortcut-targets-by-id\12r7NVnUc9cMtuw-IqvGIc2uNipmfyODe\Cases\20029'
