import deepimpact


def main():
    #######################
    #   Airburst Solver   #
    #######################

    # Initialise the Planet class
    earth = deepimpact.Planet()

    # Solve the atmospheric entry problem for a given set of input parameters
    result = earth.solve_atmospheric_entry(radius=35, angle=45,
                                           strength=1e7, density=3000,
                                           velocity=19e3)

    # Calculate the kinetic energy lost per unit altitude and add it
    # as a column to the result dataframe
    result = earth.calculate_energy(result)

    # Determine the outcomes of the impact event
    outcome = earth.analyse_outcome(result)

    #####################
    #   Damage Mapper   #
    #####################

    # Calculate the blast location and damage radius
    # for several pressure levels
    pressures = [1e3, 3.5e3, 27e3, 43e3]

    bearing = 217.  # Bearing of the meteoroid trajectory

    blast_lat, blast_lon, damage_rad = deepimpact.damage_zones(
        outcome, lat=55.2, lon=-2.5, bearing=bearing, pressures=pressures
        )

    # Plot a circle to show the limit of the lowest damage level
    damage_map = deepimpact.plot_circle(blast_lat, blast_lon, damage_rad[0])
    damage_map.save("damage_map.html")

    # The GeospatialLocator tool
    locator = deepimpact.GeospatialLocator()

    # Find the postcodes in the damage radii
    postcodes = locator.get_postcodes_by_radius((blast_lat, blast_lon),
                                                radii=damage_rad)

    # Find the population in each postcode
    population = locator.get_population_by_radius((blast_lat, blast_lon),
                                                  radii=damage_rad)

    # Print the number of people affected in each damage zone
    print()
    print("Pressure |      Damage | Population")
    print("   (kPa) | radius (km) |   affected")
    print("-----------------------------------")
    for pop, rad, zone in zip(population, damage_rad, pressures):
        print(f"{zone/1e3:8.0f} | {rad/1e3:11.1f} | {pop:10,.0f}")
    print()

    # Print the postcodes inside the highest damage zone
    print("Postcodes in the highest damage zone:")
    print(*postcodes[-1])
    print()

    # Example usage of impact_risk function
    # Uses the default file impact_parameter_list.csv in the resources folder
    risk_probability, risk_population = deepimpact.impact_risk(earth,
                                                               pressure=27e3)
    print(risk_probability.head())
    print("Total population affected: " +
          f"{risk_population['mean']:,.0f} " +
          f"+/- {risk_population['stdev']:,.0f}")

    ############################
    # Extension functionality  #
    ############################

    # Here we will calculate damage ellipses, and then postcodes,
    # populations and risk for elliptical damage zones.

    a_perp, a_front, a_behind = deepimpact.damage_ellipses(outcome, damage_rad,
                                                           kf=0.4, kb=0.2)

    postcodes_e = locator.get_postcodes_by_ellipse((blast_lat, blast_lon),
                                                   a_perp=a_perp,
                                                   a_front=a_front,
                                                   a_behind=a_behind,
                                                   bearing=bearing)

    population_e = locator.get_population_by_ellipse((blast_lat, blast_lon),
                                                     a_perp=a_perp,
                                                     a_front=a_front,
                                                     a_behind=a_behind,
                                                     bearing=bearing)

    # Print the number of people affected in each damage zone
    print()
    print("Pressure |      Damage | Pop affected | Pop affected")
    print("   (kPa) | radius (km) |     (circle) |    (ellipse)")
    print("----------------------------------------------------")
    for pop, pop_e, rad, zone in zip(population, population_e,
                                     damage_rad, pressures):
        print(f"{zone/1e3:8.0f} | {rad/1e3:11.1f} | " +
              f"{pop:12,.0f} | {pop_e:12,.0f}")
    print()

    # Print the postcodes inside the highest damage zone
    print("Postcodes in the highest elliptical damage zone:")
    print(*postcodes_e[-1])
    print()

    risk_probability_e, risk_population_e = deepimpact.impact_risk(
        earth, pressure=27e3, ellipse=True, kf=0.4, kb=0.2
        )

    print(risk_probability_e.head())
    print("Total population affected (ellipse): " +
          f"{risk_population_e['mean']:,.0f} " +
          f"+/- {risk_population_e['stdev']:,.0f}")


if __name__ == "__main__":
    main()
