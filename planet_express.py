''' 
School project - Základy počítačové fyziky 1, 2018/2019, Václav Maixner

Target of this simulation is to use the Euler's method for solving ODEs
to simulate Earth's and Moon's orbiting around Sun. Testing scenarios were
chosen as follows:
    1)  Earth completes elipsoidal motion around Sun with the duration of 12
        months and finishes as close to the starting point as possible (to prevent
        slewing on larger time scale) with the same energy.
    2)  Moon orbits around the Earth monthly, observable by 12 peaks in kinetic energy.
        Again, we expect Moon to finish it's orbit at the same position and the same energy.

The output is a png, containing position overview during the year, kinetic energy  during
the year and in the last row, the last 300 values of kinetic energy in detail, to help with
fine tuning the initial input parametres. A more advanced approach would be to write an 
algorithm, that tries to minimize the height of the step - however, it is not vital to get
intersection, but to get the best overlap of the eriodic function. This is left as a future
improvement.

The initial parametres were first chosen as average value and manually changed to make the
kinetic energy difference after year converge to a small enough value. These final values are
closer to the perihelion values, but for perihelion values the system diverges. This could be
because of the fact that Moon's vector of velocity in perihelion might not be exactly perpendicular
to the Earth-Sun axis.

'''

import math
import matplotlib.pyplot as plt
import numpy as np
import config
import uuid

plt.style.use('seaborn-white')

class Planet():
    def __init__(self, name, x, y, vx, vy, ax, ay, mass, E_kin, E_pot):
        self.name = name
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.mass = mass
        self.E_kin = E_kin
        self.E_pot = E_pot
        self.total_energy = 0.0
        self.x_positions=[]
        self.y_positions=[]
        self.energies=[]
        self.kin_energies=[]
        self.pot_energies=[]

    def get_velocity(self):
        return math.sqrt(self.vx**2 + self.vy**2)

    def get_force(self):
        return self.mass * math.sqrt(self.ax**2 + self.ay**2)


def create_snapshots(planets):
    snapshots = []
    for planet in planets:
        snapshot = planet.name + ', x=' + str(planet.x) + ', y=' + str(planet.y) + \
         ', vx=' +str(planet.vx) + ', vy=' + str(planet.vy) +  ', m=' + str(planet.mass) \
         + '\n'
        snapshots.append(snapshot)
    
    return snapshots


def setup_planets():
    ##(name, x, y, vx, vy, ax, ay, mass, E_kin, E_pot)
    planets = []

    earth = Planet('earth', 0.0, 149.5999e9, 29760.84, 0.0, 0.0, 0.0, 5.972e24, 0.0, 0.0)
    moon = Planet('moon', 0.0, 149.21512e9, 2.8725e4, 0.0, 0.0, 0.0, 7.342e22, 0.0, 0.0)
    sun = Planet('sun', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9885e30, 0.0, 0.0)

    planets.extend([earth, moon, sun])

    return planets


def loop_euler(planets, dt):
    GRAV_CONST = 6.67408e-11

    for planet in planets:
        for other_planet in planets:
            if not planet==other_planet:
                dx = planet.x - other_planet.x
                dy = planet.y - other_planet.y

                distance = math.sqrt(dx**2 + dy**2)

                acc_factor = -GRAV_CONST * other_planet.mass / float(distance**3)

                planet.ax += acc_factor*dx
                planet.ay += acc_factor*dy

                planet.E_pot += -GRAV_CONST * planet.mass * other_planet.mass / float(distance)

        planet.vx += planet.ax * dt
        planet.vy += planet.ay * dt

        planet.ax = 0.0
        planet.ay = 0.0

        planet.E_kin = 0.5 * planet.mass * (planet.get_velocity())**2
        planet.total_energy = planet.E_kin + planet.E_pot
        planet.energies.append(planet.total_energy)
        planet.kin_energies.append(planet.E_kin)
        planet.pot_energies.append(planet.E_pot)

    for planet in planets:
        planet.x += planet.vx * dt
        planet.y += planet.vy * dt

        planet.x_positions.append(planet.x)
        planet.y_positions.append(planet.y)

    return planets


def Main():
    dt = 1000
    year = 60*60*24*365
    planets = setup_planets()
    snapshots = create_snapshots(planets)
    print(snapshots)

    time = 0.0
    while time <= year:
        loop_euler(planets, dt)
        time += dt

    nrows = 3
    ncols = 2

    plt.subplots(nrows, ncols, figsize = (18,10))

    plt.subplot(nrows,ncols,1)
    plt.title(snapshots[0])
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.plot(planets[0].x_positions, planets[0].y_positions, 'tab:green')

    plt.subplot(nrows,ncols,2)
    plt.title(snapshots[1])
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.plot(planets[1].x_positions, planets[1].y_positions, 'tab:gray')
    
    plt.subplot(nrows,ncols,3)
    plt.xlabel('time', fontsize=14)
    plt.ylabel('E_kin', fontsize=14)
    plt.plot(planets[0].kin_energies, 'tab:red')

    plt.subplot(nrows,ncols,4)
    plt.xlabel('time', fontsize=14)
    plt.ylabel('E_kin', fontsize=14)
    plt.plot(planets[1].kin_energies, 'tab:brown')

    no_elements = 300
    earth_ext_kin_energies = planets[0].kin_energies[-no_elements:]
    earth_ext_kin_energies.extend(planets[0].kin_energies[:no_elements])

    moon_ext_kin_energies = planets[1].kin_energies[-no_elements:]
    moon_ext_kin_energies.extend(planets[1].kin_energies[:no_elements])

    plt.subplot(nrows,ncols,5)
    plt.xlabel('time', fontsize=14)
    plt.ylabel('E_kin', fontsize=14)
    plt.plot(earth_ext_kin_energies, 'tab:red')

    plt.subplot(nrows,ncols,6)
    plt.xlabel('time', fontsize=14)
    plt.ylabel('E_kin', fontsize=14)
    plt.plot(moon_ext_kin_energies, 'tab:brown')
    
    plt.savefig(uuid.uuid4().hex + '.png')
    plt.show()

Main()




