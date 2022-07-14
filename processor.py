import astropy.table as atpy
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import astropy.coordinates as acoo


def mccon_finder(tab, name):
    return np.array([_.rstrip() for _ in tab['GalaxyName']]) == name


mv_sun = 4.2


def matched_filter_range(xs, cen, sig, minv, maxv, ntarg):
    """
    Find the best range to detect the second population among the
    dataset that consists of xs and the second population can
    be represented as N(cen, sig) and will have ntarg objects
    
    The method returns the 'best' range 
    
    """
    binsize = 0.1
    hh, loc = np.histogram(xs,
                           range=[minv, maxv],
                           bins=int((maxv - minv) / binsize))
    xloc = loc[:-1] + .5 * np.diff(loc)
    N = scipy.stats.norm(cen, sig)
    hh1 = N.pdf(xloc)
    rat = hh1 / np.maximum(hh, 1)
    ratmax = rat.max()
    thresholds = np.linspace(0, ratmax, 20)
    logps = np.zeros(len(thresholds))
    edges1 = logps * 0
    edges2 = logps * 0

    for i, curt in enumerate(thresholds):
        xind = rat > curt
        if xind.sum() == 0:
            continue
        xind = np.nonzero(xind)[0]
        x1, x2 = xind[0], xind[-1]
        left = loc[x1]
        right = loc[x2 + 1]
        ncontam = ((xs > left) & (xs < right)).sum()
        nobj = ntarg * (N.cdf(right) - N.cdf(left))
        logp = scipy.stats.poisson(ncontam).logsf(nobj + ncontam)
        # this is the logp of background + object data given
        # background only model (i.e. significance)
        logps[i] = logp
        edges1[i] = left
        edges2[i] = right
    pos = np.argmin(logps)
    return (edges1[pos], edges2[pos])


def get_feh_mean_sig(log10l):
    """ 
    Return the expected mean metallicity and spread at a given 
    log10(L)
    """
    fehmean = -1.68 + .29 * (log10l - 6)
    fehsig = 0.45 - 0.06 * (log10l - 5)
    # mass met rela  from simon 2019
    # feh = -1.68 + .29 * (log10(L/Lsun)-6)
    # mdf width from
    # https://iopscience.iop.org/article/10.1088/0004-637X/727/2/78
    return fehmean, fehsig


def trier(fehs, mv_parent, min_mv=-14, max_mv=0, nbins=40, rstate=None):
    """
    given the array of metallicities and luminosity of the system 
    return the possible (upper limit) on number of accreted systems
    of different luminosities
    
    Returns:
    mv_grid: array of M_Vs for accreted systems
    n_16: 16% percentile on the number of accreted systems
    n_84: 84%

    """
    if rstate is None:
        rstate = np.random.default_rng()

    feh_mean_spread_massmet = 0.15  # spread in mass metallicity relation
    lfeh_sig_spread_massmet = 0.1  # spread in ln(sigma) # correspond to spread of 0.05 at sig=0.5

    nstars0 = len(fehs)
    log10l_parent = (mv_sun - mv_parent) / 2.5
    mv_sat_grid = np.linspace(min_mv, max_mv, nbins)
    nums1 = np.zeros_like(mv_sat_grid)
    nums2 = np.zeros_like(mv_sat_grid)

    nspread = 100
    nsim = 1000

    for i, mv_sat in enumerate(mv_sat_grid):
        log10l_sat = (-(mv_sat - mv_sun) / 2.5)
        rat = 10**(log10l_sat - log10l_parent)
        fehmean_sat, fehsig_sat = get_feh_mean_sig(log10l_sat)
        expn0_sat = (rat * nstars0)
        # expected number of stars from the satellite in our data

        # print('x', propfehmean)
        # figure the optimal metallicity range
        feh_left, feh_right = matched_filter_range(fehs, fehmean_sat,
                                                   fehsig_sat, -4, 1,
                                                   expn0_sat)
        nobs = ((fehs < feh_right) & (fehs > feh_left)).sum()
        sims = []
        for j in range(nspread):
            curfehmean = fehmean_sat + feh_mean_spread_massmet * rstate.normal(
            )
            curfehsig = fehsig_sat * np.exp(
                lfeh_sig_spread_massmet * rstate.normal())
            N = scipy.stats.norm(curfehmean, curfehsig)
            # this is the number is expected in the metallicity range
            expn_sat = expn0_sat * (N.cdf(feh_right) - N.cdf(feh_left))
            # print(mv_sat, feh_left, feh_right, expn0_sat, expn_sat, nobs)

            # here we write the likelihood for a
            # in N_obs ~ Poisson(a * expn_sat)
            G = scipy.stats.gamma(nobs + 1, scale=1. / expn_sat)
            G.rstate = rstate
            sims.append(G.rvs(nsim))
        sims = np.concatenate(sims)
        n_16, n_84 = [scipy.stats.scoreatpercentile(sims, _) for _ in [16, 84]]
        # n_16, n_84 = G.ppf(.16), G.ppf(.84)
        # print(propexpn, nobs, n1, en1)
        nums1[i] = n_16
        nums2[i] = n_84
    return mv_sat_grid, nums1, nums2


if __name__ == '__main__':
    tab = atpy.Table().read('kirby2010.vot')
    # https://ui.adsabs.harvard.edu/?#abs/2010ApJS..191..352K
    xt = atpy.Table().read('mcconnachie_jan2021.fits')
    C = acoo.SkyCoord(ra=xt['RA'], dec=xt['Dec'], unit=['hour', 'deg'])
    xt['ra'] = C.ra.deg
    xt['dec'] = C.dec.deg

    # name correspondence
    maps = [('CanesVenatici(1)', 'CVnI'), ('Draco', 'Dra'), ('Fornax', 'For'),
            ('Leo1', 'LeoI'), ('Leo2', 'LeoII'), ('Sculptor', 'Scl'),
            ('Sextans(1)', 'Sex'), ('UrsaMinor', 'UMi')]

    rh = xt['rh'] / 60  # in deg
    MV = xt['Vmag'] - xt['dmod']
    log10l = 10**(-1. / 2.5 * (MV - mv_sun))
    fehs = tab['__Fe_H_']
    cnt = 0
    rstate = np.random.default_rng(43435226)
    plt.clf()
    for n_mc, n_kirb in maps:
        # xind1 = xt['GalaxyName']==n_mc
        xind1 = mccon_finder(xt, n_mc)
        curmv = float(MV[xind1])
        curlogl = float(log10l[xind1])

        print(curmv)
        xind2 = tab['dSph'] == n_kirb
        assert (xind2.sum() > 0)

        xmv, xn1, xn2 = trier(fehs[xind2], curmv, rstate=rstate)
        plt.subplot(3, 3, cnt + 1)
        plt.title(n_kirb)
        plt.fill_between(xmv, xn1, xn2)
        plt.axvline(curmv, color='red')
        plt.axhline(1, linestyle='--', color='red')
        plt.gca().set_yscale('log')
        if cnt > 6:
            plt.xlabel('$M_V$')
        if cnt % 3 == 0:
            plt.ylabel('Max(N$_{merged}$)')
        cnt += 1
        plt.ylim(0.1, 100)
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig('dwarf_dwarf.pdf')
