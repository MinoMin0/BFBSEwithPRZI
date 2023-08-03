# The Bristol Frequent Batch Stock Exchange (BFBSE)

[![Python version: 3.4+](https://img.shields.io/badge/python-3.4+-blue.svg)](https://www.python.org/download/releases/3.4.0/)
[![](https://img.shields.io/github/issues/MichaelRol/Threaded-Bristol-Stock-Exchange)](https://github.com/MichaelRol/Threaded-Bristol-Stock-Exchange/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

BFBSE is an extension of Dave Cliff's (University of Bristol) [Bristol Stock Exchange (BSE)](https://github.com/davecliff/BristolStockExchange "Bristol Stock Exchange") limit order-book financial exchange simulator and Rollins' [Threaded Bristol Stock Exchange (TBSE)](https://github.com/MichaelRol/Threaded-Bristol-Stock-Exchange "Threaded Bristol Stock Exchange"). BFBSE aims to act as a minimal simulation of a limit-order-book financial exchange which uses Frequent Batch Auctions. By using a first of its kind simulator of a FBA, I  evaluated the effectiveness of various trading algorithms under the constraints of discrete time intervals and batch processing, which they have not previously been tested with. 

This simulator was produced for my MEng Individual Project.

## Usage

BFBSE can be run in three different ways. These are the three different ways to enter the trader schedule. The trader schedule is the number of each type of trader present in the market session. It should be noted that in BFBSE the buyer schedule is always equal to the seller schedule, i.e. there are the same number of buyers of each type as there are sellers. So if your schedule is 5 GDX and 5 AA, that means you will have 5 GDX buyers, 5 AA buyers, 5 GDX sellers and 5 AA sellers for a total of 20 traders. There are 6 traders available in BFBSE, these are ZIC, ZIP, Giveaway, Shaver, AA, and GDX. The three ways to specify this schedule are:

#### - From the config file:

```console
$ python3 tbse.py
```
By entering no command-line arguments BFBSE will use the order schedule as it exists in ```config.py```(lines 16-21).

#### - From the command-line:

```console
$ python3 tbse.py [zic],[zip],[gdx],[aa],[gvwy],[shvr]
```
Where each trader name is replaced with the number of that trader you want in the market schedule. For example:
```console
$ python3 tbse.py 0,0,5,5,0,0
```
will produce a trader schedule with 5 GDX buyers, 5 GDX sellers, 5 AA buyers and 5 AA sellers. You must enter a number for each of the 6 trader types, so put 0 if you do not want a certain trader present in your market session.

#### - From a CSV file:

```console
$ python3 tbse.py filename.csv
```

Using a CSV file is the most versatile way to use TBSE as it allows multiple market sessions to be defined using different trader schedules. TBSE will run each row of the CSV file as a separate market session. Each row must contain 6 comma-separated numbers in the order ZIC, ZIP, GDX, AA, Giveaway, Shaver, so 0 should be used if you don't wish a trader to be present in a market session. The following example CSV file will run experiments comparing 5 vs 5 of every possible pair of traders:

```
5, 5, 0, 0, 0, 0
5, 0, 5, 0, 0, 0
5, 0, 0, 5, 0, 0
5, 0, 0, 0, 5, 0
5, 0, 0, 0, 0, 5
0, 5, 5, 0, 0, 0
0, 5, 0, 5, 0, 0
0, 5, 0, 0, 5, 0
0, 5, 0, 0, 0, 5
0, 0, 5, 5, 0, 0
0, 0, 5, 0, 5, 0
0, 0, 5, 0, 0, 5
0, 0, 0, 5, 5, 0
0, 0, 0, 5, 0, 5
0, 0, 0, 0, 5, 5
```

## Config

Market sessions ran in BFBSE can be configured by editing ```config.py```. It should be noted that lines 60 onwards are for verifying the content of the configuration file and should not be changed. These lines will alert the user if they have misconfigured TBSE. 

The comments within the config file should be enough for a user to understand how to configure BFBSE, any missing information should be found in the [BSE Guide](https://github.com/davecliff/BristolStockExchange/blob/master/BSEguide1.2e.pdf "BSE Guide") which describes things like the different stepmodes and timemodes available. 

## License
The code is open-sourced via the [MIT](http://opensource.org/licenses/mit-license.php) Licence: see the LICENSE file for full text. 
