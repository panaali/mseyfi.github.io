Here’s a Python example that walks through **Object-Oriented Programming (OOP) principles** using **three classes** and highlights all the concepts you mentioned.

We’ll create a small system around **Vehicles**, including a base class and two subclasses.

```python
from abc import ABC, abstractmethod  # For abstraction

# 1. ABSTRACTION: Define an abstract base class
class Vehicle(ABC):
    # Class attribute shared among all instances
    vehicle_count = 0

    def __init__(self, brand, speed):
        # 2. ENCAPSULATION: Private attributes with double underscore
        self.__brand = brand
        self.__speed = speed
        Vehicle.vehicle_count += 1

    # 3. GETTERS AND SETTERS with property decorators
    @property
    def brand(self):
        return self.__brand

    @brand.setter
    def brand(self, value):
        self.__brand = value

    @property
    def speed(self):
        return self.__speed

    @speed.setter
    def speed(self, value):
        if value < 0:
            raise ValueError("Speed cannot be negative")
        self.__speed = value

    # 4. PRIVATE METHOD
    def __update_log(self):
        print("Private: Vehicle log updated")

    # 5. ABSTRACT METHOD (force subclasses to implement this)
    @abstractmethod
    def drive(self):
        pass

    # 6. STATIC METHOD (doesn't access class or instance)
    @staticmethod
    def convert_kmph_to_mph(kmph):
        return kmph * 0.621371

    # 7. CLASS METHOD (access class-level data)
    @classmethod
    def get_vehicle_count(cls):
        return cls.vehicle_count


# 8. INHERITANCE: Car inherits from Vehicle
class Car(Vehicle):
    def __init__(self, brand, speed, model):
        super().__init__(brand, speed)  # Call the base constructor
        self.__model = model

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    # 9. POLYMORPHISM: Implementing abstract method differently
    def drive(self):
        return f"Driving car {self.brand} {self.model} at {self.speed} km/h"


# Another subclass
class Motorcycle(Vehicle):
    def __init__(self, brand, speed, has_sidecar):
        super().__init__(brand, speed)
        self.__has_sidecar = has_sidecar

    @property
    def has_sidecar(self):
        return self.__has_sidecar

    # 10. POLYMORPHISM continued
    def drive(self):
        sidecar = "with sidecar" if self.__has_sidecar else "without sidecar"
        return f"Riding motorcycle {self.brand} {sidecar} at {self.speed} km/h"


# --- USAGE EXAMPLE ---
car1 = Car("Toyota", 120, "Camry")
bike1 = Motorcycle("Harley", 80, True)

print(car1.drive())        # Polymorphism
print(bike1.drive())       # Polymorphism

print(Vehicle.get_vehicle_count())  # Class method
print(Vehicle.convert_kmph_to_mph(100))  # Static method

car1.speed = 150  # Setter usage
print(car1.speed)  # Getter usage
```

---

### 🔍 Concepts Explained

| Concept | Explanation |
|--------|-------------|
| **Encapsulation** | Wrapping data (attributes) with methods to protect them. E.g., `__brand` is private and accessed via getter/setter. |
| **Abstraction** | Hiding implementation details. The abstract method `drive()` must be implemented in subclasses. |
| **Polymorphism** | Subclasses provide different implementations of `drive()`. |
| **Inheritance** | `Car` and `Motorcycle` inherit from `Vehicle`. |
| **Setters/Getters** | Using `@property` and `@<name>.setter` to manage access and validation. |
| **Private Attributes** | `__brand` and `__speed` are private, only accessible via methods. |
| **Private Methods** | `__update_log()` is only usable inside the class. |
| **Abstract Method vs Private Method** | `@abstractmethod` enforces subclass behavior; `__method()` is just hidden for internal use. |
| **Class Attributes** | `vehicle_count` tracks all vehicles, shared across instances. |
| **Static Method** | Utility like unit conversion; doesn’t access instance or class (`convert_kmph_to_mph`). |
| **Class Method** | Operates on the class, used here to access shared state (`get_vehicle_count`). |

---

Let me know if you want the same explanation with an example in another domain, like employees or animals — or refactored in a more functional way.
