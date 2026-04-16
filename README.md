A collection of routines to:

(1) perform photometry with TESS (and Kepler/K2) target pixel files, using a user-selectable range of algorithms, 
as well as detrending the resulting light curves against CCD position and (optionally) background level. Most algorithms implement different
ways of determining the photometric aperture to use (AOP = aperture-optimized photometry), and some can accommodate multiple targets in the field
of view. Output from TESSCut is comprehensible as input. Codes can be run via a GUI overlay.

(2) Simultaneous analysis of photometric light curves and contemporaneous ground-based polarimetry, in order to detect and characterize oscillation modes. 
Input light curve format required is consistent with output from (1) above, but a tool is available to convert native TESS SPOC light curves into that 
format. 
