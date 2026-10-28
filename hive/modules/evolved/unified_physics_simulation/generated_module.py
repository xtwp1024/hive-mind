class Vector:
    """A class to represent a two-dimensional vector."""
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        """Add two vectors."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Subtract two vectors."""
        return Vector(self.x - other.x, self.y - other.y)

    def dot(self, other):
        """Calculate the dot product of two vectors."""
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """Calculate the cross product of two vectors."""
        return self.x * other.y - self.y * other.x

    def magnitude(self):
        """Calculate the magnitude of the vector."""
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __str__(self):
        return f"({self.x}, {self.y})"


class Particle:
    """A class to represent a particle with position and velocity."""
    
    def __init__(self, position=Vector(), velocity=Vector()):
        self.position = position
        self.velocity = velocity

    def update(self, time_step):
        """Update the particle's position based on its velocity."""
        self.position = self.position + self.velocity * time_step

    def __str__(self):
        return f"Particle at {self.position}"


class UnifiedPhysicsSimulationManager:
    """A class to manage physics simulations."""
    
    def __init__(self):
        self.particles = []

    def add_particle(self, particle):
        """Add a particle to the simulation."""
        if not isinstance(particle, Particle):
            raise ValueError("Only instances of Particle can be added.")
        self.particles.append(particle)

    def simulate(self, time_step, duration):
        """Simulate the physics over a specified duration."""
        for _ in range(int(duration / time_step)):
            for particle in self.particles:
                particle.update(time_step)

    def get_particles(self):
        """Get the list of particles in the simulation."""
        return self.particles


# Unit tests for the classes
import unittest

class TestVector(unittest.TestCase):
    def test_add(self):
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        self.assertEqual(v1 + v2, Vector(4, 6))

    def test_sub(self):
        v1 = Vector(5, 6)
        v2 = Vector(3, 4)
        self.assertEqual(v1 - v2, Vector(2, 2))

    def test_dot(self):
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        self.assertEqual(v1.dot(v2), 11)

    def test_cross(self):
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        self.assertEqual(v1.cross(v2), -2)

    def test_magnitude(self):
        v1 = Vector(3, 4)
        self.assertEqual(v1.magnitude(), 5)


class TestParticle(unittest.TestCase):
    def test_update(self):
        particle = Particle(Vector(1, 2), Vector(1, 1))
        particle.update(1)
        self.assertEqual(particle.position, Vector(2, 3))


class TestUnifiedPhysicsSimulationManager(unittest.TestCase):
    def test_add_particle(self):
        manager = UnifiedPhysicsSimulationManager()
        particle = Particle(Vector(1, 2), Vector(1, 1))
        manager.add_particle(particle)
        self.assertIn(particle, manager.particles)

    def test_simulate(self):
        manager = UnifiedPhysicsSimulationManager()
        particle = Particle(Vector(1, 2), Vector(1, 1))
        manager.add_particle(particle)
        manager.simulate(1, 1)
        self.assertEqual(particle.position, Vector(2, 3))

    def test_get_particles(self):
        manager = UnifiedPhysicsSimulationManager()
        particle = Particle(Vector(1, 2), Vector(1, 1))
        manager.add_particle(particle)
        self.assertEqual(manager.get_particles(), [particle])


if __name__ == '__main__':
    unittest.main()