# Object-Oriented Programming (OOP) principles

We’ll create a small system around **Vehicles**, including a base class and two subclasses.

```python
from abc import ABC, abstractmethod  # For abstraction

# 1. ABSTRACTION: Define an abstract base class
class Vehicle(ABC):
    # Class/static attribute is shared among all instances. If it is changed anywhere, its value across all the instances will also change accordingly.
    # It is created once for all the instances.
    # It is good when we want to share properties/values among all instances.


    vehicle_count = 0

    def __init__(self, brand, speed):
        # 2. ENCAPSULATION: Private attributes with double underscore
        self.__brand = brand
        self.__speed = speed
        Vehicle.vehicle_count += 1
        # if we use self. then here only the count will be incremented in the instance and will not reflect to other instances.
        # self.vegicle_count += 1

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

    # 6. STATIC METHOD (doesn't access class or instance (it cannot access self.), is shared between all the objects, we can access it on class level or instance level)
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

Encapsulation is **super important** in object-oriented programming because it’s all about **control, safety, and clarity**. Here's why it's such a big deal:

---

### 🛡️ 1. **Data Protection**
- By **hiding internal variables** (e.g., using `__private_var`), you prevent other parts of the program from messing with the internal state of your object in unexpected or invalid ways.
- For example, if you had a `speed` attribute, you don't want someone to set it to `-100`.

```python
car.speed = -100  # Without encapsulation, this might go unnoticed and break logic!
```

---

### ✅ 2. **Validation and Logic Control**
- Encapsulation lets you add **logic to setters**, so you can **validate or transform** data when it's set.
- For instance, with `@property`:

```python
@speed.setter
def speed(self, value):
    if value < 0:
        raise ValueError("Speed must be positive")
    self.__speed = value
```

---

### 🧼 3. **Cleaner Interfaces**
- The outside world doesn't need to know how your class works internally. They just use your **methods or properties**.
- This makes your code **easier to understand and maintain** — less clutter, fewer mistakes.

---

### 🔄 4. **Easier Refactoring**
- You can **change how something is stored internally** without affecting code that uses your class, as long as the interface (methods, properties) stays the same.

---

### 🔐 5. **Enforces Boundaries**
- Helps **modularize your code**: each class takes care of its own data and behavior, and other parts of the system just interact with it through safe methods.
- It's like saying: "Here’s what you can do, and here’s what you **shouldn’t** touch."

---

### 🧠 TL;DR:
Encapsulation helps you write **robust, secure, and clean code** by:
- Hiding sensitive data
- Validating inputs
- Exposing only what’s necessary
- Reducing bugs and accidental misuse

---
## Protected vs Private
```python
class Base:
    def __init__(self):
        self._protected = "Protected"
        self.__private = "Private"

class Sub(Base):
    def access(self):
        print(self._protected)       # ✅ allowed
        # print(self.__private)     # ❌ not allowed
        print(self._Base__private)   # ⚠️ works, but yikes
```
