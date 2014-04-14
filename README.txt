This is a quick test of a TRBM microworld
We will set up a 2D maze and wake-sleep train an RBM
Then try annealed Subiculum inference on it


PLAN

-Make 10-loc plus maze (or use ICEA)
-Each loc has: 3 whiskers, one light-ahead sensor, one color-background sensor
-HD cells

-Make dict[x,y,th] of raw senses (by hand?)
	x,y = -2,-1,0,1,2
	th = [0,1], [0,-1] etc

-Generate histories by random walks (stay,fwd,turn)
      -set light sensor depending on state?
   -light sequence advances when arrives at on lights

   -include DG: combis of:
	- HD&light_ahead
	- whiskerL&whiskerR
        - whiskerL&whiskerR&whiskerF

-CA3: fix combis (from noiseless EC/DG data)
	place
	place&HD
	light_seq
	place&light_seq

-Wake-sleep, on fixed CA3 combis (trains difficult conjunction weights).  Allow -ve weights.
        first, WS-train CA3 only (no inputs). 
	       these will give approximate prdictions, due to inherent unpredictability of the path
     add noise to inputs, WS-train DG-CA3 for that noise level
	    add noise in a sensible way
	    	eg. ensure exactly one grid and HD always on
		REFACTOR: add noise to EC then recompute DG??
     train obs weights, using ground truth CA3 ??? what about bias?-- own separate bias then fuse?
	[some say no learning in DG-CA3, only in EC-DG, but ignore this]

-Do annealed inference
    	     - no decoding needed yet -- just look at CA3 place cells directly



- test improvement given by fused bel, how much error now?
     - show bel|ground_CA3  (at present we're using the previous CA3_hat)

     - vs pi|ground_CA3

     - vs |CA3_hat -- show lostness?

     - lam only?

- Decode positions?   (maybe enforce magic inhibition to force once PC on at all times?)
     - or do that in an EC atttractor?  need to train new weights back to EC, +bias.
      NB only for place, there are no CA3 pure HD cells

- Resurrect ICEA graphs of estimated position

- Tweak params to allow lostness to occur (eg more obs noise)

>>>

-Add Subiculum
     disable priors / or raise their temperature at least

- run on much larger data set to get light transitions



TODO Need to think about how resets will work in any Ach\propto Sub version?

TODO need to show some proper loss of tracking -- keeps gettign lucky on return this way?
     maybe bigger plus maze?  (use old ICEA code to generate)
     TODO: as in ICEA version, deliberately kidnap to another arm to show track failure.


NB remember that the sub can be triggered by the EC obs being wrong, when the CA3 prediction is right!
   that's ok.
   its only when Sub is triggered for a sustained time that we think something is wrong

*TODO noisy GPS updates --> feed place+HD tracking back into EC, add odom
      assumes that previous estimate was correct.
