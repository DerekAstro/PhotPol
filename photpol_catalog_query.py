#!/usr/bin/env python3
"""Private, disposable catalog worker for PhotPol; called by the extractor."""
import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('service', choices=['gaia', 'simbad'])
    parser.add_argument('ra', type=float)
    parser.add_argument('dec', type=float)
    parser.add_argument('radius', type=float, help='arcmin for Gaia, arcsec for SIMBAD')
    parser.add_argument('output')
    args = parser.parse_args()
    # Imports, TAP metadata discovery, submission, polling, and result retrieval
    # all happen here and are covered by the parent's wall-clock deadline.
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    coord = SkyCoord(ra=args.ra * u.deg, dec=args.dec * u.deg)
    if args.service == 'gaia':
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = -1
        table = Gaia.cone_search_async(coordinate=coord, radius=args.radius*u.arcmin).get_results()
    else:
        from astroquery.simbad import Simbad
        sim = Simbad()
        sim.add_votable_fields('ids')
        table = sim.query_region(coord, radius=args.radius*u.arcsec)
    with Path(args.output).open('wb') as stream:
        pickle.dump(table, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
