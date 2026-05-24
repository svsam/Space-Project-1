from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

planets = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="""
            pl_name,
            hostname,
            
            sy_dist,

            pl_orbper,
            pl_orbsmax,
            pl_bmasse,
            pl_rade,
            pl_rvamp,

            pl_orbeccen,
            pl_orbeccenerr1,
            pl_orbeccenerr2,
            pl_orbeccenlim,
            
            st_mass,
            st_lum,

            discoverymethod"""
)

df = planets.to_pandas()
df["st_lum_solar"] = 10 **df["st_lum"]
zero_e = df[df["pl_orbeccen"] == 0]

print(zero_e[[ "pl_name",
        "hostname",
        "st_mass",
        "st_lum_solar",
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "pl_orbeccenerr1", 
        "pl_orbeccenerr2",
        "pl_orbeccenlim"
]].head(20))

df.to_csv("exoplanets.csv", index=False)

