"""Module containing all trader algos"""
# pylint: disable=too-many-lines
import math
import random
import sys

from tbse_msg_classes import Order
from tbse_sys_consts import TBSE_SYS_MAX_PRICE, TBSE_SYS_MIN_PRICE, TICK_SIZE
import time as time1


# pylint: disable=too-many-instance-attributes
class Trader:
    """Trader superclass - mostly unchanged from original BSE code by Dave Cliff
    all Traders have a trader id, bank balance, blotter, and list of orders to execute"""

    def __init__(self, ttype, tid, balance, params, time):
        self.ttype = ttype      # what type / strategy this trader is
        self.tid = tid          # trader unique ID code
        self.balance = balance  # money in the bank
        self.params = params    # parameters/extras associated with this trader-type or individual trader.        
        self.blotter = []       # record of trades executed
        self.blotter_length = 100   # maximum length of blotter
        self.orders = {}  # customer orders currently being worked (fixed at 1)
        self.n_quotes = 0  # number of quotes live on LOB
        self.willing = 1  # used in ZIP etc
        self.able = 1  # used in ZIP etc
        self.birth_time = time  # used when calculating age of a trader/strategy
        self.n_trades = 0  # how many trades has this trader done?
        self.profitpertime = 0      # profit per unit time
        self.last_quote = None  # record of what its last quote was
        self.times = [0, 0, 0, 0]  # values used to calculate timing elements
    def __str__(self):
        return f'[TID {self.tid} type {self.ttype} balance {self.balance} blotter {self.blotter} ' \
               f'orders {self.orders} n_trades {self.n_trades} profit_per_time {self.profit_per_time}]'

    def add_order(self, order, verbose):
        """
        Adds an order to the traders list of orders
        in this version, trader has at most one order,
        if allow more than one, this needs to be self.orders.append(order)
        :param order: the order to be added
        :param verbose: should verbose logging be printed to console
        :return: Response: "Proceed" if no current offer on LOB, "LOB_Cancel" if there is an order on the LOB needing
                 cancelled.\
        """

        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders[order.coid] = order

        if verbose:
            print(f'add_order < response={response}')
        return response

    def del_order(self, coid):
        """
        Removes current order from traders list of orders
        :param coid: Customer order ID of order to be deleted
        """
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders.pop(coid)

    def bookkeep(self, trade, order, verbose, time):
        """
        Updates trader's internal stats with trade and order
        :param trade: Trade that has been executed
        :param order: Order trade was in response to
        :param verbose: Should verbose logging be printed to console
        :param time: Current time
        """
        output_string = ""

        if trade['coid'] in self.orders:
            coid = trade['coid']
            order_price = self.orders[coid].price
        elif trade['counter'] in self.orders:
            coid = trade['counter']
            order_price = self.orders[coid].price
        else:
            print("COID not found")
            sys.exit("This is non ideal ngl.")

        self.blotter.append(trade)  # add trade record to trader's blotter
        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transaction_price = trade['price']
        if self.orders[coid].otype == 'Bid':
            profit = order_price - transaction_price
        else:
            profit = transaction_price - order_price
        self.balance += profit
        self.n_trades += 1
        self.profit_per_time = self.balance / (time - self.birth_time)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            print(str(trade['coid']) + " " + str(trade['counter']) + " " + str(order.coid) + " " + str(
                self.orders[0].coid))
            sys.exit()

        if verbose:
            print(f'{output_string} profit={profit} balance={self.balance} profit/t={self.profit_per_time}')
        self.del_order(coid)  # delete the order

    # pylint: disable=unused-argument,no-self-use
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        specify how trader responds to events in the market
        this is a null action, expect it to be overloaded by specific algos
        :param time: Current time
        :param lob: Limit order book
        :param trade: Trade being responded to
        :param verbose: Should verbose logging be printed to console
        :return: Unused
        """
        return None

    # pylint: disable=unused-argument,no-self-use
   
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Get's the traders order based on the current state of the market
        :param time: Current time
        :param countdown: Time to end of session
        :param lob: Limit order book
        :return: The order
        """
        return None


class TraderGiveaway(Trader):
    """
    Trader subclass Giveaway
    even dumber than a ZI-U: just give the deal away
    (but never makes a loss)
    """
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Get's giveaway traders order - in this case the price is just the limit price from the customer order
        :param time: Current time
        :param countdown: Time until end of session
        :param lob: Limit order book
        :return: Order to be sent to the exchange
        """

        if len(self.orders) < 1:
            order = None
        else:
            coid = max(self.orders.keys())
            quote_price = self.orders[coid].price
            order = Order(self.tid,
                          self.orders[coid].otype,
                          quote_price,
                          self.orders[coid].qty,
                          time, self.orders[coid].coid, self.orders[coid].toid)
            self.last_quote = order
            #print(f"Trader {self.tid} of {self.orders[coid].otype} has orders with limit prices {[o[1].price for o in self.orders.items()]} at time {time} \n")
       
        return order


class TraderZic(Trader):
    """ Trader subclass ZI-C
    After Gode & Sunder 1993"""
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Gets ZIC trader, limit price is randomly selected
        :param time: Current time
        :param countdown: Time until end of current market session
        :param lob: Limit order book
        :return: The trader order to be sent to the exchange
        """

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            
            coid = max(self.orders.keys())
            
            min_price_lob = lob['bids']['worst']
            max_price_lob = lob['asks']['worst']
            limit = self.orders[coid].price
            otype = self.orders[coid].otype

            min_price = min_price_lob
            max_price = max_price_lob

            if otype == 'Bid':
                if min_price>limit:
                    min_price=min_price_lob
                quote_price = random.randint(min_price, limit)
            else:
                if max_price<limit:
                    max_price=max_price_lob
                quote_price = random.randint(limit, max_price)
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order    
        
        return order


class TraderShaver(Trader):
    """Trader subclass Shaver
    shaves a penny off the best price
    if there is no best price, creates "stub quote" at system max/min"""
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Get's Shaver trader order by shaving/adding a penny to current best bid
        :param time: Current time
        :param countdown: Countdown to end of market session
        :param lob: Limit order book
        :return: The trader order to be sent to the exchange
        """
        if len(self.orders) < 1:
            order = None
        else:

            coid = max(self.orders.keys())
            limit_price = self.orders[coid].price
            otype = self.orders[coid].otype

            best_bid = 500
            best_ask = 0
            
            if demand_curve!=[]:
                best_bid = max(demand_curve, key=lambda x: x[0])[0]+1

            if supply_curve!=[]:
                best_ask = min(supply_curve, key=lambda x: x[0])[0]-1    

            if otype == 'Bid':
                quote_price= best_bid
                quote_price = min(quote_price, limit_price)
            else:
                quote_price = best_ask
                quote_price = max(quote_price, limit_price)

            #quote_price = min(quote_price, limit_price)
            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order

        return order


class TraderSniper(Trader):
    """
    Trader subclass Sniper
    Based on Shaver,
    "lurks" until t remaining < threshold% of the trading session
    then gets increasing aggressive, increasing "shave thickness" as t runs out"""
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        :param time: Current time
        :param countdown: Time until end of market session
        :param lob: Limit order book
        :return: Trader order to be sent to exchange
        """
        lurk_threshold = 0.2
        shave_growth_rate = 3
        shave = int(1.0 / (0.01 + countdown / (shave_growth_rate * lurk_threshold)))
        if (len(self.orders) < 1) or (countdown > lurk_threshold):
            order = None
        else:
            coid = max(self.orders.keys())
            limit_price = self.orders[coid].price
            otype = self.orders[coid].otype

            if demand_curve!=None and supply_curve!=None:

                best_bid = min(demand_curve, key=lambda x: x[0])[0]
                best_ask = max(supply_curve, key=lambda x: x[0])[0]
            else:
                best_bid = lob['bids']['worst'] - shave
                best_ask = lob['asks']['worst'] + shave


            if otype == 'Bid':
                    quote_price = best_bid+shave
                    quote_price = min(quote_price, limit_price)    

            else:
                    quote_price = best_ask-shave
                    quote_price = max(quote_price, limit_price)  
            
            quote_price = min(quote_price, limit_price)    
            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order
        return order


# Trader subclass ZIP
# After Cliff 1997
# pylint: disable=too-many-instance-attributes
class TraderZip(Trader):
    """ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    NB this implementation keeps separate margin values for buying & selling,
       so a single trader can both buy AND sell
       -- in the original, traders were either buyers OR sellers"""

    def __init__(self, ttype, tid, balance, time):

        Trader.__init__(self, ttype, tid, balance, time)
        m_fix = 0.05
        m_var = 0.3
        self.job = None  # this is 'Bid' or 'Ask' depending on customer order
        self.active = False  # gets switched to True while actively working an order
        self.prev_change = 0  # this was called last_d in Cliff'97
        self.beta = 0.2 + 0.2 * random.random()  # learning rate #0.1 + 0.2 * random.random()
        self.momentum = 0.3 * random.random()  # momentum #0.3 * random.random()
        self.ca = 0.10  # self.ca & .cr were hard-coded in '97 but parameterised later
        self.cr = 0.10
        self.margin = None  # this was called profit in Cliff'97
        self.margin_buy = -1.0 * (m_fix + m_var * random.random())
        self.margin_sell = m_fix + m_var * random.random()
        self.price = None
        self.limit = None
        self.times = [0, 0, 0, 0]
        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None
        self.last_batch = None

    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        :param time: Current time
        :param countdown: Time until end of current market session
        :param lob: Limit order book
        :return: Trader order to be sent to exchange
        """
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            coid = max(self.orders.keys())
            self.active = True
            self.limit = self.orders[coid].price
            self.job = self.orders[coid].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quote_price = int(self.limit * (1 + self.margin))
            self.price = quote_price

            order = Order(self.tid, self.job, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order
        return order

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        update margin on basis of what happened in marke
        ZIP trader responds to market events, altering its margin
        does this whether it currently has an order to work or not
        :param time: Current time
        :param lob: Limit order book
        :param trade: Trade being responded to
        :param verbose: Should verbose logging be printed to console
        """
        
        if self.last_batch==(demand_curve,supply_curve):
            return
        else:
            self.last_batch = (demand_curve,supply_curve)
     
        trade = trades[0] if trades else None
    
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve!=[]:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve!=[]:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]

        def target_up(price):
            """
            generate a higher target price by randomly perturbing given price
            :param price: Current price
            :return: New price target
            """
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))

            return target

        def target_down(price):
            """
            generate a lower target price by randomly perturbing given price
            :param price: Current price
            :return: New price target
            """
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))

            return target

        def willing_to_trade(price):
            """
            am I willing to trade at this price?
            :param price: Price to be traded out
            :return: Is the trader willing to trade
            """
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            """
            Update target profit margin
            :param price: New target profit margin
            """
            old_price = self.price
            diff = price - old_price
            change = ((1.0 - self.momentum) * (self.beta * diff)) + (self.momentum * self.prev_change)
            self.prev_change = change
            new_margin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if new_margin < 0.0:
                    self.margin_buy = new_margin
                    self.margin = new_margin
            else:
                if new_margin > 0.0:
                    self.margin_sell = new_margin
                    self.margin = new_margin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False

        #lob_best_bid_p = lob['bids']['best']
        lob_best_bid_p = best_bid #CHANGE HERE
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            elif self.prev_best_bid_p < lob_best_bid_p:
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1] #might have to check if has been cancelled at some point during batch 
            #for item in lob['tape'] check if cancel happened with price of
            if last_tape_item['type'] == 'Cancel': 
                #print("Last bid was cancelled") #test
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        #lob_best_ask_p = lob['asks']['best']
        lob_best_ask_p = best_ask #CHANGE HERE
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            elif self.prev_best_ask_p > lob_best_ask_p:
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                #print("Last bid was cancelled") # test
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if trade is None:
            deal = False

        if self.job == 'Ask':
            # seller
            if deal:
                trade_price = trade['price']
                if self.price <= trade_price:
                    # could sell for more? raise margin
                    target_price = target_up(trade_price)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(trade_price):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_down(trade_price)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                trade_price = trade['price']
                if self.price >= trade_price:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(trade_price)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(trade_price):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_up(trade_price)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q


# pylint: disable=too-many-instance-attributes
class TraderAa(Trader):
    """
    Daniel Snashall's implementation of Vytelingum's AA trader, first described in his 2006 PhD Thesis.
    For more details see: Vytelingum, P., 2006. The Structure and Behaviour of the Continuous Double
    Auction. PhD Thesis, University of Southampton
    """

    def __init__(self, ttype, tid, balance, time):
        # Stuff about trader
        super().__init__(ttype, tid, balance, time)
        self.active = False

        self.limit = None
        self.job = None

        # learning variables
        self.r_shout_change_relative = 0.05
        self.r_shout_change_absolute = 0.05
        self.short_term_learning_rate = random.uniform(0.1, 0.5)
        self.long_term_learning_rate = random.uniform(0.1, 0.5)
        self.moving_average_weight_decay = 0.95  # how fast weight decays with t, lower is quicker, 0.9 in vytelingum
        self.moving_average_window_size = 5
        self.offer_change_rate = 3.0
        self.theta = -2.0
        self.theta_max = 2.0
        self.theta_min = -8.0
        self.market_max = TBSE_SYS_MAX_PRICE

        # Variables to describe the market
        self.previous_transactions = []
        self.moving_average_weights = []
        for i in range(self.moving_average_window_size):
            self.moving_average_weights.append(self.moving_average_weight_decay ** i)
        self.estimated_equilibrium = []
        self.smiths_alpha = []
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        # Trading Variables
        self.r_shout = None
        self.buy_target = None
        self.sell_target = None
        self.buy_r = -1.0 * (0.3 * random.random())
        self.sell_r = -1.0 * (0.3 * random.random())

        #define last batch so that internal values are only updated upon new batch matching
        self.last_batch = None

    def calc_eq(self):
        """
        Calculates the estimated 'eq' or estimated equilibrium price.
        Slightly modified from paper, it is unclear in paper
        N previous transactions * weights / N in Vytelingum, swap N denominator for sum of weights to be correct?
        :return: Estimated equilibrium price
        """
        if len(self.previous_transactions) == 0:
            return
        if len(self.previous_transactions) < self.moving_average_window_size:
            # Not enough transactions
            self.estimated_equilibrium.append(
                float(sum(self.previous_transactions)) / max(len(self.previous_transactions), 1))
        else:
            n_previous_transactions = self.previous_transactions[-self.moving_average_window_size:]
            thing = [n_previous_transactions[i] * self.moving_average_weights[i] for i in
                     range(self.moving_average_window_size)]
            eq = sum(thing) / sum(self.moving_average_weights)
            self.estimated_equilibrium.append(eq)

    def calc_alpha(self):
        """
        Calculates trader's alpha value - see AA paper for details.
        """
        alpha = 0.0
        for p in self.estimated_equilibrium:
            alpha += (p - self.estimated_equilibrium[-1]) ** 2
        alpha = math.sqrt(alpha / len(self.estimated_equilibrium))
        self.smiths_alpha.append(alpha / self.estimated_equilibrium[-1])

    def calc_theta(self):
        """
        Calculates trader's theta value - see AA paper for details.
        """
        gamma = 2.0  # not sensitive apparently so choose to be whatever
        # necessary for initialisation, div by 0
        if min(self.smiths_alpha) == max(self.smiths_alpha):
            alpha_range = 0.4  # starting value i guess
        else:
            alpha_range = (self.smiths_alpha[-1] - min(self.smiths_alpha)) / (
                    max(self.smiths_alpha) - min(self.smiths_alpha))
        theta_range = self.theta_max - self.theta_min
        desired_theta = self.theta_min + theta_range * (1 - (alpha_range * math.exp(gamma * (alpha_range - 1))))
        self.theta = self.theta + self.long_term_learning_rate * (desired_theta - self.theta)

    def calc_r_shout(self):
        """
        Calculates trader's r shout value - see AA paper for details.
        """
        p = self.estimated_equilibrium[-1]
        lim = self.limit
        theta = self.theta
        if self.job == 'Bid':
            # Currently a buyer
            if lim <= p:  # extra-marginal!
                self.r_shout = 0.0
            else:  # intra-marginal :(
                if self.buy_target > self.estimated_equilibrium[-1]:
                    # r[0,1]
                    self.r_shout = math.log(((self.buy_target - p) * (math.exp(theta) - 1) / (lim - p)) + 1) / theta
                else:
                    # r[-1,0]
                    self.r_shout = math.log((1 - (self.buy_target / p)) * (math.exp(theta) - 1) + 1) / theta

        if self.job == 'Ask':
            # Currently a seller
            if lim >= p:  # extra-marginal!
                self.r_shout = 0
            else:  # intra-marginal :(
                if self.sell_target > self.estimated_equilibrium[-1]:
                    # r[-1,0]
                    self.r_shout = math.log(
                        (self.sell_target - p) * (math.exp(theta) - 1) / (self.market_max - p) + 1) / theta
                else:
                    # r[0,1]
                    a = (self.sell_target - lim) / (p - lim)
                    self.r_shout = (math.log((1 - a) * (math.exp(theta) - 1) + 1)) / theta

    def calc_agg(self):
        """
        Calculates Trader's aggressiveness parameter - see AA paper for details.
        """
        if self.job == 'Bid':
            # BUYER
            if self.buy_target >= self.previous_transactions[-1]:
                # must be more aggressive
                delta = (1 + self.r_shout_change_relative) * self.r_shout + self.r_shout_change_absolute
            else:
                delta = (1 - self.r_shout_change_relative) * self.r_shout - self.r_shout_change_absolute

            self.buy_r = self.buy_r + self.short_term_learning_rate * (delta - self.buy_r)

        if self.job == 'Ask':
            # SELLER
            if self.sell_target > self.previous_transactions[-1]:
                delta = (1 + self.r_shout_change_relative) * self.r_shout + self.r_shout_change_absolute
            else:
                delta = (1 - self.r_shout_change_relative) * self.r_shout - self.r_shout_change_absolute

            self.sell_r = self.sell_r + self.short_term_learning_rate * (delta - self.sell_r)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def calc_target(self):
        """
        Calculates trader's target price - see AA paper for details.
        """
        p = 1
        if len(self.estimated_equilibrium) > 0:
            p = self.estimated_equilibrium[-1]
            if self.limit == p:
                p = p * 1.000001  # to prevent theta_bar = 0
        elif self.job == 'Bid':
            p = self.limit - self.limit * 0.2  # Initial guess for eq if no deals yet!!....
        elif self.job == 'Ask':
            p = self.limit + self.limit * 0.2
        lim = self.limit
        theta = self.theta
        if self.job == 'Bid':
            # BUYER
            minus_thing = (math.exp(-self.buy_r * theta) - 1) / (math.exp(theta) - 1)
            plus_thing = (math.exp(self.buy_r * theta) - 1) / (math.exp(theta) - 1)
            theta_bar = (theta * lim - theta * p) / p
            if theta_bar == 0:
                theta_bar = 0.0001
            if math.exp(theta_bar) - 1 == 0:
                theta_bar = 0.0001
            bar_thing = (math.exp(-self.buy_r * theta_bar) - 1) / (math.exp(theta_bar) - 1)
            if lim <= p:  # Extra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim
                else:
                    self.buy_target = lim * (1 - minus_thing)
            else:  # intra-marginal
                if self.buy_r >= 0:
                    self.buy_target = p + (lim - p) * plus_thing
                else:
                    self.buy_target = p * (1 - bar_thing)
            self.buy_target = min(self.buy_target, lim)

        if self.job == 'Ask':
            # SELLER
            minus_thing = (math.exp(-self.sell_r * theta) - 1) / (math.exp(theta) - 1)
            plus_thing = (math.exp(self.sell_r * theta) - 1) / (math.exp(theta) - 1)
            theta_bar = (theta * lim - theta * p) / p
            if theta_bar == 0:
                theta_bar = 0.0001
            if math.exp(theta_bar) - 1 == 0:
                theta_bar = 0.0001
            bar_thing = (math.exp(-self.sell_r * theta_bar) - 1) / (math.exp(theta_bar) - 1)  # div 0 sometimes what!?
            if lim <= p:  # Extra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim
                else:
                    self.buy_target = lim + (self.market_max - lim) * minus_thing
            else:  # intra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim + (p - lim) * (1 - plus_thing)
                else:
                    self.buy_target = p + (self.market_max - p) * bar_thing
            if self.sell_target is None:
                self.sell_target = lim
            elif self.sell_target < lim:
                self.sell_target = lim

    # pylint: disable=too-many-branches
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Creates an AA trader's order
        :param time: Current time
        :param countdown: Time left in the current trading period
        :param lob: Current state of the limit order book
        :return: Order to be sent to the exchange
        """
        if len(self.orders) < 1:
            self.active = False
            return None
        coid = max(self.orders.keys())
        self.active = True
        self.limit = self.orders[coid].price
        self.job = self.orders[coid].otype
        self.calc_target()

        if self.prev_best_bid_p is None:
            o_bid = 0
        else:
            o_bid = self.prev_best_bid_p
        if self.prev_best_ask_p is None:
            o_ask = self.market_max
        else:
            o_ask = self.prev_best_ask_p

        quote_price = TBSE_SYS_MIN_PRICE
        if self.job == 'Bid':  # BUYER
            if self.limit <= o_bid:
                return None
            if len(self.previous_transactions) > 0:  # has been at least one transaction
                o_ask_plus = (1 + self.r_shout_change_relative) * o_ask + self.r_shout_change_absolute
                quote_price = o_bid + ((min(self.limit, o_ask_plus) - o_bid) / self.offer_change_rate)
            else:
                if o_ask <= self.buy_target:
                    quote_price = o_ask
                else:
                    quote_price = o_bid + ((self.buy_target - o_bid) / self.offer_change_rate)
        elif self.job == 'Ask':
            if self.limit >= o_ask:
                return None
            if len(self.previous_transactions) > 0:  # has been at least one transaction
                o_bid_minus = (1 - self.r_shout_change_relative) * o_bid - self.r_shout_change_absolute
                quote_price = o_ask - ((o_ask - max(self.limit, o_bid_minus)) / self.offer_change_rate)
            else:
                if o_bid >= self.sell_target:
                    quote_price = o_bid
                else:
                    quote_price = o_ask - ((o_ask - self.sell_target) / self.offer_change_rate)

        order = Order(self.tid, self.job, int(quote_price), self.orders[coid].qty, time, self.orders[coid].coid,
                      self.orders[coid].toid)
        self.last_quote = order
        return order

    # pylint: disable=too-many-branches
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        Updates AA trader's internal variables based on activities on the LOB
        Beginning nicked from ZIP
        what, if anything, has happened on the bid LOB? Nicked from ZIP.
        :param time: current time
        :param lob: current state of the limit order book
        :param trade: trade which occurred to trigger this response
        :param verbose: should verbose logging be printed to the console
        """

        if self.last_batch==(demand_curve,supply_curve):
            return
        else:
            self.last_batch = (demand_curve,supply_curve)
     
        trade = trades[0] if trades else None
    
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve!=[]:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve!=[]:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]

        bid_hit = False
        #lob_best_bid_p = lob['bids']['best'] #CHANGED
        lob_best_bid_p = best_bid
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            # elif self.prev_best_bid_p < lob_best_bid_p :
            #     # best bid has improved
            #     # NB doesn't check if the improvement was by self
            #     bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        # ask_improved = False
        ask_lifted = False

        #lob_best_ask_p = lob['asks']['best'] #CHANGED THIS
        lob_best_ask_p = best_ask
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            # elif self.prev_best_ask_p > lob_best_ask_p :
            #     # best ask has improved -- NB doesn't check if the improvement was by self
            #     ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

        deal = bid_hit or ask_lifted
        if (trades==[]):
            deal = False

        # End nicked from ZIP

        if deal:
            # if trade is not None:
            self.previous_transactions.append(trade['price'])
            if self.sell_target is None:
                self.sell_target = trade['price'] #CHANGED THIS
                #self.sell_target = best_ask
            if self.buy_target is None:
                self.buy_target = trade['price'] #CHANGED THIS
                #self.sell_target = best_bid
            self.calc_eq()
            self.calc_alpha()
            self.calc_theta()
            self.calc_r_shout()
            self.calc_agg()
            self.calc_target()


# pylint: disable=too-many-instance-attributes
class TraderGdx(Trader):
    """
    Daniel Snashall's implementation of Tesauro & Bredin's GDX Trader algorithm. For more details see:
    Tesauro, G., Bredin, J., 2002. Sequential Strategic Bidding in Auctions using Dynamic Programming.
    Proceedings AAMAS2002.
    """
    def __init__(self, ttype, tid, balance, time):
        super().__init__(ttype, tid, balance, time)
        self.prev_orders = []
        self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
        self.active = False  # gets switched to True while actively working an order
        self.limit = None

        # memory of all bids and asks and accepted bids and asks
        self.outstanding_bids = []
        self.outstanding_asks = []
        self.accepted_asks = []
        self.accepted_bids = []

        self.price = -1

        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        self.first_turn = True

        self.gamma = 0.9

        self.holdings = 25
        self.remaining_offer_ops = 25
        self.values = [[0 for _ in range(self.remaining_offer_ops)] for _ in range(self.holdings)]

        #define last batch so that internal values are only updated upon new batch matching
        self.last_batch = None

    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Creates a GDX trader's order
        :param time: Current time
        :param countdown: Time left in the current trading period
        :param lob: Current state of the limit order book
        :return: Order to be sent to the exchange
        """
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            coid = max(self.orders.keys())
            self.active = True
            self.limit = self.orders[coid].price
            self.job = self.orders[coid].otype

            # calculate price
            if self.job == 'Bid':
                self.price = self.calc_p_bid(self.holdings - 1, self.remaining_offer_ops - 1)
            if self.job == 'Ask':
                self.price = self.calc_p_ask(self.holdings - 1, self.remaining_offer_ops - 1)

            order = Order(self.tid, self.job, int(self.price), self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order

        if self.first_turn or self.price == -1:
            return None
        return order

    def calc_p_bid(self, m, n):
        """
        Calculates the price the GDX trader should bid at. See GDX paper for more details.
        :param m: Table of expected values
        :param n: Remaining opportunities to make an offer
        :return: Price to bid at
        """
        best_return = 0
        best_bid = 0
        # second_best_return = 0
        second_best_bid = 0

        # first step size of 1 get best and 2nd best
        for i in [x * 2 for x in range(int(self.limit / 2))]:
            thing = self.belief_buy(i) * ((self.limit - i) + self.gamma * self.values[m - 1][n - 1]) + (
                    1 - self.belief_buy(i) * self.gamma * self.values[m][n - 1])
            if thing > best_return:
                second_best_bid = best_bid
                # second_best_return = best_return
                best_return = thing
                best_bid = i

        # always best bid largest one
        if second_best_bid > best_bid:
            a = second_best_bid
            second_best_bid, best_bid = best_bid, a
            # second_best_bid = best_bid
            # best_bid = a

        # then step size 0.05
        for i in [x * 0.05 for x in range(int(second_best_bid), int(best_bid))]:
            thing = self.belief_buy(i + second_best_bid) * (
                    (self.limit - (i + second_best_bid)) + self.gamma * self.values[m - 1][n - 1]) + (
                            1 - self.belief_buy(i + second_best_bid) * self.gamma * self.values[m][n - 1])
            if thing > best_return:
                best_return = thing
                best_bid = i + second_best_bid

        return best_bid

    def calc_p_ask(self, m, n):
        """
        Calculates the price the GDX trader should sell at. See GDX paper for more details.
        :param m: Table of expected values
        :param n: Remaining opportunities to make an offer
        :return: Price to sell at
        :return: Price to sell at
        """
        best_return = 0
        best_ask = self.limit
        # second_best_return = 0
        second_best_ask = self.limit

        # first step size of 1 get best and 2nd best
        for i in [x * 2 for x in range(int(self.limit / 2))]:
            j = i + self.limit
            thing = self.belief_sell(j) * ((j - self.limit) + self.gamma * self.values[m - 1][n - 1]) + (
                    1 - self.belief_sell(j) * self.gamma * self.values[m][n - 1])
            if thing > best_return:
                second_best_ask = best_ask
                # second_best_return = best_return
                best_return = thing
                best_ask = j
        # always best ask largest one
        if second_best_ask > best_ask:
            a = second_best_ask
            second_best_ask, best_ask = best_ask, a
            # second_best_ask = best_ask
            # best_ask = a

        # then step size 0.05
        for i in [x * 0.05 for x in range(int(second_best_ask), int(best_ask))]:
            thing = self.belief_sell(i + second_best_ask) * (
                    ((i + second_best_ask) - self.limit) + self.gamma * self.values[m - 1][n - 1]) + (
                            1 - self.belief_sell(i + second_best_ask) * self.gamma * self.values[m][n - 1])
            if thing > best_return:
                best_return = thing
                best_ask = i + second_best_ask

        return best_ask

    def belief_sell(self, price):
        """
        Calculates the 'belief' that a certain price will be accepted and traded on the exchange.
        :param price: The price for which we want to calculate the belief.
        :return: The belief value (decimal).
        """
        accepted_asks_greater = 0
        bids_greater = 0
        unaccepted_asks_lower = 0
        for p in self.accepted_asks:
            if p >= price:
                accepted_asks_greater += 1
        for p in [thing[0] for thing in self.outstanding_bids]:
            if p >= price:
                bids_greater += 1
        for p in [thing[0] for thing in self.outstanding_asks]:
            if p <= price:
                unaccepted_asks_lower += 1

        if accepted_asks_greater + bids_greater + unaccepted_asks_lower == 0:
            return 0
        return (accepted_asks_greater + bids_greater) / (accepted_asks_greater + bids_greater + unaccepted_asks_lower)

    def belief_buy(self, price):
        """
        Calculates the 'belief' that a certain price will be accepted and traded on the exchange.
        :param price: The price for which we want to calculate the belief.
        :return: The belief value (decimal).
        """
        accepted_bids_lower = 0
        asks_lower = 0
        unaccepted_bids_greater = 0
        for p in self.accepted_bids:
            if p <= price:
                accepted_bids_lower += 1
        for p in [thing[0] for thing in self.outstanding_asks]:
            if p <= price:
                asks_lower += 1
        for p in [thing[0] for thing in self.outstanding_bids]:
            if p >= price:
                unaccepted_bids_greater += 1
        if accepted_bids_lower + asks_lower + unaccepted_bids_greater == 0:
            return 0
        return (accepted_bids_lower + asks_lower) / (accepted_bids_lower + asks_lower + unaccepted_bids_greater)

    def get_best_n_bids(self,demand_curve, n):
        bids = []
        last_item_count = 0
        for price, quantity in demand_curve:
            num_bids = quantity-last_item_count
            last_item_count = quantity
            bids += [price] * num_bids
            if len(bids) >= n:
                return bids[:n]
        return bids

    def get_best_n_asks(self,supply_curve, n):
        asks = []
        last_item_count = 0
        for price, quantity in reversed(supply_curve):
            num_asks = quantity-last_item_count
            last_item_count = quantity
            asks += [price] * num_asks
            if len(asks) >= n:
                return asks[:n]
        return asks
    
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        Updates GDX trader's internal variables based on activities on the LOB
        :param time: current time
        :param lob: current state of the limit order book
        :param trade: trade which occurred to trigger this response
        :param verbose: should verbose logging be printed to the console
        """
        if self.last_batch==(demand_curve,supply_curve):
            return
        else:
            self.last_batch = (demand_curve,supply_curve)
            # print(f"demand_curve {demand_curve}")
            # print(f"supply curve {supply_curve}")
     
        trade = trades[0] if trades else None
    
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve!=[]:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve!=[]:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]
        
        # what, if anything, has happened on the bid LOB?
        self.outstanding_bids = lob['bids']['lob']
        # bid_improved = False
        # bid_hit = False
        # lob_best_bid_p = lob['bids']['best']
        lob_best_bid_p = best_bid
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            # elif self.prev_best_bid_p < lob_best_bid_p :
            #     # best bid has improved
            #     # NB doesn't check if the improvement was by self
            #     bid_improved = True

            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit

                #self.accepted_bids.append(self.prev_best_bid_p) #CHANGED HERE
                self.accepted_bids.extend(self.get_best_n_bids(demand_curve,q_eq))
                #self.accepted_bids.extend([p for p,q in demand_curve[:q_eq]])
                #print(f"adding {[p for p,q in demand_curve[:q_eq]]}")
                # bid_hit = True
        # elif self.prev_best_bid_p is not None:
        #     # the bid LOB has been emptied: was it cancelled or hit?
        #     last_tape_item = lob['tape'][-1]
        # if last_tape_item['type'] == 'Cancel' :
        #     bid_hit = False
        # else:
        #     bid_hit = True

        # what, if anything, has happened on the ask LOB?
        self.outstanding_asks = lob['asks']['lob']
        # ask_improved = False
        # ask_lifted = False
        #lob_best_ask_p = lob['asks']['best']
        lob_best_ask_p = best_ask
        lob_best_ask_q = None

        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            # elif self.prev_best_ask_p > lob_best_ask_p :
            # best ask has improved -- NB doesn't check if the improvement was by self
            # ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                #self.accepted_asks.append(self.prev_best_ask_p) #CHANGED THIS
                self.accepted_asks.extend(self.get_best_n_asks(supply_curve,q_eq))
                #self.accepted_asks.extend([p for p,q in supply_curve[-q_eq:]])
                #print(f"adding {[p for p,q in supply_curve[-q_eq:]]}")

                # ask_lifted = True
        # elif self.prev_best_ask_p is not None:
        # the ask LOB is empty now but was not previously: canceled or lifted?
        # last_tape_item = lob['tape'][-1]
        # if last_tape_item['type'] == 'Cancel' :
        #     ask_lifted = False
        # else:
        #     ask_lifted = True

        # populate expected values
        if self.first_turn:
            self.first_turn = False
            for n in range(1, self.remaining_offer_ops):
                for m in range(1, self.holdings):
                    if self.job == 'Bid':
                        # BUYER
                        self.values[m][n] = self.calc_p_bid(m, n)

                    if self.job == 'Ask':
                        # BUYER
                        self.values[m][n] = self.calc_p_ask(m, n)

        # deal = bid_hit or ask_lifted

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

# pylint: disable=too-many-instance-attributes
class Trader_PRZI(Trader):

    # how to mutate the strategy values when evolving / hill-climbing
    def mutate_strat(self, s, mode):
        s_min = self.strat_range_min
        s_max = self.strat_range_max
        if mode == 'gauss':
            sdev = 0.05
            newstrat = s
            while newstrat == s:
                newstrat = s + random.gauss(0.0, sdev)
                # truncate to keep within range
                newstrat = max(-1.0, min(1.0, newstrat))
        elif mode == 'uniform_whole_range':
            # draw uniformly from whole range
            newstrat = random.uniform(-1.0, +1.0)
        elif mode == 'uniform_bounded_range':
            # draw uniformly from bounded range
            newstrat = random.uniform(s_min, s_max)
        else:
            sys.exit('FAIL: bad mode in mutate_strat')
        return newstrat


    def strat_str(self):
        # pretty-print a string summarising this trader's strategies
        string = '%s: %s active_strat=[%d]:\n' % (self.tid, self.ttype, self.active_strat)
        for s in range(0, self.k):
            strat = self.strats[s]
            stratstr = '[%d]: s=%+f, start=%f, $=%f, pps=%f\n' % \
                       (s, strat['stratval'], strat['start_t'], strat['profit'], strat['pps'])
            string = string + stratstr

        return string


    def __init__(self, ttype, tid, balance, params, time):
        # if params == "landscape-mapper" then it generates data for mapping the fitness landscape

        verbose = True

        Trader.__init__(self, ttype, tid, balance, params, time)

        # unpack the params
        # for all three of PRZI, PRSH, and PRDE params can include strat_min and strat_max
        # for PRSH and PRDE params should include values for optimizer and k
        # if no params specified then defaults to PRZI with strat values in [-1.0,+1.0]

        # default parameter values
        k = 1
        optimizer = None # no optimizer => plain non-adaptive PRZI
        s_min = -1.0
        s_max = +1.0
        
        # did call provide different params?
        if type(params) is dict:
            if 'k' in params:
                k = params['k']
            if 'optimizer' in params:
                optimizer = params['optimizer']
            s_min = params['strat_min']
            s_max = params['strat_max']
        
        self.optmzr = optimizer     # this determines whether it's PRZI, PRSH, or PRDE
        self.k = k                  # number of sampling points (cf number of arms on a multi-armed-bandit, or pop-size)
        self.theta0 = 100           # threshold-function limit value
        self.m = 4                  # tangent-function multiplier
        self.strat_wait_time = 7200     # how many secs do we give any one strat before switching?
        self.strat_range_min = s_min    # lower-bound on randomly-assigned strategy-value
        self.strat_range_max = s_max    # upper-bound on randomly-assigned strategy-value
        self.active_strat = 0       # which of the k strategies are we currently playing? -- start with 0
        self.prev_qid = None        # previous order i.d.
        self.strat_eval_time = self.k * self.strat_wait_time   # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.profit_epsilon = 0.0 * random.random()    # minimum profit-per-sec difference between strategies that counts
        self.strats = []            # strategies awaiting initialization
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1,10))  # multiplier coefficient when estimating p_max
        self.mapper_outfile = None
        # differential evolution parameters all in one dictionary
        self.diffevol = {'de_state': 'active_s0',          # initial state: strategy 0 is active (being evaluated)
                         's0_index': self.active_strat,    # s0 starts out as active strat
                         'snew_index': self.k,             # (k+1)th item of strategy list is DE's new strategy
                         'snew_stratval': None,            # assigned later
                         'F': 0.8                          # differential weight -- usually between 0 and 2
                         }

        start_time = time
        profit = 0.0
        profit_per_second = 0
        lut_bid = None
        lut_ask = None

        for s in range(self.k + 1):
            # initialise each of the strategies in sequence: 
            # for PRZI: only one strategy is needed
            # for PRSH, one random initial strategy, then k-1 mutants of that initial strategy
            # for PRDE, use draws from uniform distbn over whole range and a (k+1)th strategy is needed to hold s_new
            if s == 0:
                strategy = random.uniform(self.strat_range_min, self.strat_range_max)
            else:
                if self.optmzr == 'PRSH':
                    # simple stochastic hill climber: cluster other strats around strat_0
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'gauss')     # mutant of strats[0]
                elif self.optmzr == 'PRDE':
                    # differential evolution: seed initial strategies across whole space
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'uniform_bounded_range')
                else:
                    # PRZI -- do nothing
                    pass
            self.strats.append({'stratval': strategy, 'start_t': start_time,
                                'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})
            if self.optmzr is None:
                # PRZI -- so we stop after one iteration
                break
            elif self.optmzr == 'PRSH' and s == self.k - 1:
                # PRSH -- doesn't need the (k+1)th strategy
                break

        if self.params == 'landscape-mapper':
            # replace seed+mutants set of strats with regularly-spaced strategy values over the whole range
            self.strats = []
            strategy_delta = 0.01
            strategy = -1.0
            k = 0
            self.strats = []
            while strategy <= +1.0:
                self.strats.append({'stratval': strategy, 'start_t': start_time,
                                    'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})
                k += 1
                strategy += strategy_delta
            self.mapper_outfile = open('landscape_map.csv', 'w')
            self.k = k
            self.strat_eval_time = self.k * self.strat_wait_time

        if verbose:
            print("%s\n" % self.strat_str())


    def getorder(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + TICK_SIZE  # BSE ticksize is global var
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - TICK_SIZE   # BSE ticksize is global var
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            # print('shvr_p=%f; ' % shvr_p)
            return shvr_p


        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):
            # set parameter values and calculate CDF LUT
            # strat is strategy-value in [-1,+1]
            # t0 and m are constants used in the threshold function
            # dirn is direction: 'buy' or 'sell'
            # pmin and pmax are bounds on discrete-valued price-range

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1*theta0, min(theta0, x))
                return t

            epsilon = 0.000001 #used to catch DIV0 errors
            verbose = False

            if (strat > 1.0) or (strat < -1.0):
                # out of range
                sys.exit('PRSH FAIL: strat=%f out of range\n' % strat)

            if (dirn != 'buy') and (dirn != 'sell'):
                # out of range
                sys.exit('PRSH FAIL: bad dirn=%s\n' % dirn)

            if pmax < pmin:
                # screwed
                sys.exit('PRSH FAIL: pmax %f < pmin %f \n' % (pmax, pmin))

            if verbose:
                print('PRSH calc_cdf_lut: strat=%f dirn=%d pmin=%d pmax=%d\n' % (strat, dirn, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the limit-price with probability 1

                if dirn == 'buy':
                    cdf = [{'price':pmax, 'cum_prob': 1.0}]
                else: # must be a sell
                    cdf = [{'price': pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                # normalize the price to proportion of its range
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif strat > 0:
                    if dirn == 'buy':
                        cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                    else:   # dirn == 'sell'
                        cal_p = (math.exp(c * (1 - p_r)) - 1.0) / e2cm1
                else:   # self.strat < 0
                    if dirn == 'buy':
                        cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                    else:   # dirn == 'sell'
                        cal_p = 1.0 - ((math.exp(c * (1 - p_r)) - 1.0) / e2cm1)

                if cal_p < 0:
                    cal_p = 0   # just in case

                calp_interval.append({'price':p, "cal_p":cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                cal_p = calp_interval[p-pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}

        verbose = False

        if verbose:
            print('t=%.1f PRSH getorder: %s, %s' % (time, self.tid, self.strat_str()))

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype
            qid = self.orders[0].qid

            if self.prev_qid is None:
                self.prev_qid = qid

            if qid != self.prev_qid:
                # customer-order i.d. has changed, so we're working a new customer-order now
                # this is the time to switch arms
                # print("New order! (how does it feel?)")
                dummy = 1

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible as defined by exchange

            # trader's individual estimate highest price the market will bear
            maxprice = self.pmax # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5) # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:        # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']         # so use that as my new estimate of highest
                    self.pmax = maxprice

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price

            strat = self.strats[self.active_strat]['stratval']

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            if otype == 'Bid':

                p_max = int(limit)
                if strat > 0.0:
                    p_min = minprice
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * minprice))

                lut_bid = self.strats[self.active_strat]['lut_bid']
                if (lut_bid is None) or \
                        (lut_bid['strat'] != strat) or\
                        (lut_bid['pmin'] != p_min) or \
                        (lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.strats[self.active_strat]['lut_bid'] = calc_cdf_lut(strat, self.theta0, self.m, 'buy', p_min, p_max)

                lut = self.strats[self.active_strat]['lut_bid']

            else:   # otype == 'Ask'

                p_min = int(limit)
                if strat > 0.0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * maxprice))
                    if p_max < p_min:
                        # this should never happen, but just in case it does...
                        p_max = p_min


                lut_ask = self.strats[self.active_strat]['lut_ask']
                if (lut_ask is None) or \
                        (lut_ask['strat'] != strat) or \
                        (lut_ask['pmin'] != p_min) or \
                        (lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.strats[self.active_strat]['lut_ask'] = calc_cdf_lut(strat, self.theta0, self.m, 'sell', p_min, p_max)

                lut = self.strats[self.active_strat]['lut_ask']

                
            verbose = False
            if verbose:
                print('PRZI strat=%f LUT=%s \n \n' % (strat, lut))
                # useful in debugging: print a table of lut: price and cum_prob, with the discrete derivative (gives PMF).
                last_cprob = 0.0
                for lut_entry in lut['cdf_lut']:
                    cprob = lut_entry['cum_prob']
                    print('%d, %f, %f' % (lut_entry['price'], cprob - last_cprob, cprob))
                    last_cprob = cprob
                print('\n');    
                
                # print ('[LUT print suppressed]')
            
            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order


    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:] # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('PRSH FAIL: negative profit')

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

        self.strats[self.active_strat]['profit'] += profit
        time_alive = time - self.strats[self.active_strat]['start_t']
        if time_alive > 0:
            profit_per_second = self.strats[self.active_strat]['profit'] / time_alive
            self.strats[self.active_strat]['pps'] = profit_per_second
        else:
            # if it trades at the instant it is born then it would have infinite profit-per-second, which is insane
            # to keep things sensible whne time_alive == 0 we say the profit per second is whatever the actual profit is
            self.strats[self.active_strat]['pps'] = profit


    # PRSH respond() asks/answers two questions
    # do we need to choose a new strategy? (i.e. have just completed/cancelled previous customer order)
    # do we need to dump one arm and generate a new one? (i.e., both/all arms have been evaluated enough)
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):    


        # "PRSH" is a very basic form of stochastic hill-climber (SHC) that's v easy to understand and to code
        # it cycles through the k different strats until each has been operated for at least eval_time seconds
        # but a strat that does nothing will get swapped out if it's been running for no_deal_time without a deal
        # then the strats with the higher total accumulated profit is retained,
        # and mutated versions of it are copied into the other k-1 strats
        # then all counters are reset, and this is repeated indefinitely
        #
        # "PRDE" uses a basic form of Differential Evolution. This maintains a population of at least four strats
        # iterates indefinitely on:
        #       shuffle the set of strats;
        #       name the first four strats s0 to s3;
        #       create new_strat=s1+f*(s2-s3);
        #       evaluate fitness of s0 and new_strat;
        #       if (new_strat fitter than s0) then new_strat replaces s0.
        #
        # todo: add in other optimizer algorithms that are cleverer than these
        #  e.g. inspired by multi-arm-bandit algos like like epsilon-greedy, softmax, or upper confidence bound (UCB)

        verbose = False

        # first update each strategy's profit-per-second (pps) value -- this is the "fitness" of each strategy
        for s in self.strats:
            # debugging check: make profit be directly proportional to strategy, no noise
            # s['profit'] = 100 * abs(s['stratval'])
            # update pps
            pps_time = time - s['start_t']
            if pps_time > 0:
                s['pps'] = s['profit'] / pps_time
            else:
                s['pps'] = s['profit']


        if self.optmzr == 'PRSH':

            if verbose:
                # print('t=%f %s PRSH respond: shc_algo=%s eval_t=%f max_wait_t=%f' %
                #     (time, self.tid, shc_algo, self.strat_eval_time, self.strat_wait_time))
                dummy = 1

            # do we need to swap strategies?
            # this is based on time elapsed since last reset -- waiting for the current strategy to get a deal
            # -- otherwise a hopeless strategy can just sit there for ages doing nothing,
            # which would disadvantage the *other* strategies because they would never get a chance to score any profit.

            # NB this *cycles* through the available strats in sequence

            s = self.active_strat
            time_elapsed = time - self.last_strat_change_time
            if time_elapsed > self.strat_wait_time:
                # we have waited long enough: swap to another strategy

                new_strat = s + 1
                if new_strat > self.k - 1:
                    new_strat = 0

                self.active_strat = new_strat
                self.last_strat_change_time = time

                if verbose:
                    print('t=%f %s PRSH respond: strat[%d] elapsed=%f; wait_t=%f, switched to strat=%d' %
                          (time, self.tid, s, time_elapsed, self.strat_wait_time, new_strat))

            # code below here deals with creating a new set of k-1 mutants from the best of the k strats

            # assume that all strats have had long enough, and search for evidence to the contrary
            all_old_enough = True
            for s in self.strats:
                lifetime = time - s['start_t']
                if lifetime < self.strat_eval_time:
                    all_old_enough = False
                    break

            if all_old_enough:
                # all strategies have had long enough: which has made most profit?

                # sort them by profit
                strats_sorted = sorted(self.strats, key = lambda k: k['pps'], reverse = True)
                # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                if verbose:
                    print('PRSH %s: strat_eval_time=%f, all_old_enough=True' % (self.tid, self.strat_eval_time))
                    for s in strats_sorted:
                        print('s=%f, start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

                if self.params == 'landscape-mapper':
                    for s in self.strats:
                        self.mapper_outfile.write('time, %f, strat, %f, pps, %f\n' %
                              (time, s['stratval'], s['pps']))
                    self.mapper_outfile.flush()
                    sys.exit()

                else:
                    # if the difference between the top two strats is too close to call then flip a coin
                    # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                    best_strat = 0
                    prof_diff = strats_sorted[0]['pps'] - strats_sorted[1]['pps']
                    if abs(prof_diff) < self.profit_epsilon:
                        # they're too close to call, so just flip a coin
                        best_strat = random.randint(0,1)

                    if best_strat == 1:
                        # need to swap strats[0] and strats[1]
                        tmp_strat = strats_sorted[0]
                        strats_sorted[0] = strats_sorted[1]
                        strats_sorted[1] = tmp_strat

                    # the sorted list of strats replaces the existing list
                    self.strats = strats_sorted

                    # at this stage, strats_sorted[0] is our newly-chosen elite-strat, about to replicate

                    # now replicate and mutate the elite into all the other strats
                    for s in range(1, self.k):    # note range index starts at one not zero (elite is at [0])
                        self.strats[s]['stratval'] = self.mutate_strat(self.strats[0]['stratval'], 'gauss')
                        self.strats[s]['start_t'] = time
                        self.strats[s]['profit'] = 0.0
                        self.strats[s]['pps'] = 0.0
                    # and then update (wipe) records for the elite
                    self.strats[0]['start_t'] = time
                    self.strats[0]['profit'] = 0.0
                    self.strats[0]['pps'] = 0.0
                    self.active_strat = 0

                if verbose:
                    print('%s: strat_eval_time=%f, MUTATED:' % (self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('s=%f start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

        elif self.optmzr == 'PRDE':
            # simple differential evolution

            # only initiate diff-evol once the active strat has been evaluated for long enough
            actv_lifetime = time - self.strats[self.active_strat]['start_t']
            if actv_lifetime >= self.strat_wait_time:

                if self.k < 4:
                    sys.exit('FAIL: k too small for diffevol')

                if self.diffevol['de_state'] == 'active_s0':
                    # we've evaluated s0, so now we need to evaluate s_new
                    self.active_strat = self.diffevol['snew_index']
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_snew'

                elif self.diffevol['de_state'] == 'active_snew':
                    # now we've evaluated s_0 and s_new, so we can do DE adaptive step
                    if verbose:
                        print('PRDE trader %s' % self.tid)
                    i_0 = self.diffevol['s0_index']
                    i_new = self.diffevol['snew_index']
                    fit_0 = self.strats[i_0]['pps']
                    fit_new = self.strats[i_new]['pps']

                    if verbose:
                        print('DiffEvol: t=%.1f, i_0=%d, i0fit=%f, i_new=%d, i_new_fit=%f' % (time, i_0, fit_0, i_new, fit_new))

                    if fit_new >= fit_0:
                        # new strat did better than old strat0, so overwrite new into strat0
                        self.strats[i_0]['stratval'] = self.strats[i_new]['stratval']

                    # do differential evolution

                    # pick four individual strategies at random, but they must be distinct
                    stratlist = list(range(0, self.k))    # create sequential list of strategy-numbers
                    random.shuffle(stratlist)             # shuffle the list

                    # s0 is next iteration's candidate for possible replacement
                    self.diffevol['s0_index'] = stratlist[0]

                    # s1, s2, s3 used in DE to create new strategy, potential replacement for s0
                    s1_index = stratlist[1]
                    s2_index = stratlist[2]
                    s3_index = stratlist[3]

                    # unpack the actual strategy values
                    s1_stratval = self.strats[s1_index]['stratval']
                    s2_stratval = self.strats[s2_index]['stratval']
                    s3_stratval = self.strats[s3_index]['stratval']

                    # this is the differential evolution "adaptive step": create a new individual
                    new_stratval = s1_stratval + self.diffevol['F'] * (s2_stratval - s3_stratval)

                    # clip to bounds
                    new_stratval = max(-1, min(+1, new_stratval))

                    # record it for future use (s0 will be evaluated first, then s_new)
                    self.strats[self.diffevol['snew_index']]['stratval'] = new_stratval

                    if verbose:
                        print('DiffEvol: t=%.1f, s0=%d, s1=%d, (s=%+f), s2=%d, (s=%+f), s3=%d, (s=%+f), sNew=%+f' %
                              (time, self.diffevol['s0_index'],
                               s1_index, s1_stratval, s2_index, s2_stratval, s3_index, s3_stratval, new_stratval))

                    # DC's intervention for fully converged populations
                    # is the stddev of the strategies in the population equal/close to zero?
                    sum = 0.0
                    for s in range(self.k):
                        sum += self.strats[s]['stratval']
                    strat_mean = sum / self.k
                    sumsq = 0.0
                    for s in range(self.k):
                        diff = self.strats[s]['stratval'] - strat_mean
                        sumsq += (diff * diff)
                    strat_stdev = math.sqrt(sumsq / self.k)
                    if verbose:
                        print('t=,%.1f, MeanStrat=, %+f, stdev=,%f' % (time, strat_mean, strat_stdev))
                    if strat_stdev < 0.0001:
                        # this population has converged
                        # mutate one srategy at random
                        randindex = random.randint(0, self.k - 1)
                        self.strats[randindex]['stratval'] = random.uniform(-1.0, +1.0)
                        if verbose:
                            print('Converged pop: set strategy %d to %+f' % (randindex, self.strats[randindex]['stratval']))

                    # set up next iteration: first evaluate s0
                    self.active_strat = self.diffevol['s0_index']
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_s0'

                else:
                    sys.exit('FAIL: self.diffevol[\'de_state\'] not recognized')

        elif self.optmzr is None:
            # this is PRZI -- nonadaptive, no optimizer, nothing to change here.
            pass

        else:
            sys.exit('FAIL: bad value for self.optmzr')
 # ----------------trader-types have all been defined now-------------
