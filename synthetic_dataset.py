"""
QuantAI Restaurant Dataset Generator
This module generates comprehensive and realistic restaurant data for training and testing the calling agent system.
Includes advanced restaurant operations, customer experience scenarios, and business workflows.
"""

import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from faker import Faker
import json
from typing import List, Dict, Any
import os

class QuantAIRestaurantDatasetGenerator:
    def __init__(self):
        """Initialize the restaurant dataset generator."""
        self.restaurant_name = "QuantAI Restaurant"
        self.owner = "Harjas Singh"
        self.fake = Faker()
        
        # Initialize core components
        self.culinary_knowledge = self._load_culinary_knowledge()
        self.infrastructure = self._initialize_restaurant_infrastructure()
        self.quality_metrics = self._initialize_quality_metrics()
        self.certifications = self._initialize_certifications()
        self.seasonal_items = self._initialize_seasonal_items()
        
        # Initialize wine and beverage program
        self.beverage_program = self._initialize_beverage_program()
        
        # Initialize supplier information
        self.suppliers = self._initialize_suppliers()
        
        # Initialize staff training programs
        self.training_programs = self._initialize_training_programs()

    def _initialize_quality_metrics(self) -> Dict[str, Any]:
        """Initialize restaurant quality metrics and performance indicators."""
        return {
            'food_safety': {
                'inspection_scores': {
                    'health_inspection': 98,
                    'food_safety_audit': 95,
                    'haccp_compliance': 97,
                    'allergen_management': 96,
                    'cross_contamination_prevention': 94
                },
                'incident_rates': {
                    'food_borne_illness': 0.001,
                    'cross_contamination': 0.002,
                    'temperature_violations': 0.003,
                    'allergen_incidents': 0.001,
                    'equipment_failures': 0.002
                },
                'preventive_measures': {
                    'daily_temperature_logs': True,
                    'weekly_cleaning_schedules': True,
                    'monthly_pest_control': True,
                    'quarterly_equipment_maintenance': True
                }
            },
            'customer_experience': {
                'satisfaction_score': 4.7,
                'wait_times': {
                    'seating': 5,  # minutes
                    'food_preparation': 15,
                    'checkout': 3,
                    'bar_service': 4,
                    'special_requests': 8
                },
                'service_quality': {
                    'staff_friendliness': 4.8,
                    'food_quality': 4.9,
                    'ambiance': 4.7,
                    'cleanliness': 4.8,
                    'menu_knowledge': 4.7,
                    'wine_knowledge': 4.6,
                    'allergen_awareness': 4.9,
                    'special_requests_handling': 4.7
                },
                'feedback_metrics': {
                    'online_reviews': 4.6,
                    'in_person_feedback': 4.8,
                    'social_media_engagement': 4.5,
                    'repeat_customer_rate': 0.65
                }
            },
            'operational_metrics': {
                'table_turnover_rate': 2.5,  # times per day
                'average_check_size': 45.00,  # dollars
                'peak_hours_occupancy': 0.85,
                'kitchen_efficiency': {
                    'prep_time': 10,  # minutes
                    'cooking_time': 15,
                    'plating_time': 3,
                    'expediting_time': 2,
                    'special_requests_time': 5
                },
                'service_efficiency': {
                    'average_table_service_time': 90,  # minutes
                    'bar_service_time': 4,  # minutes
                    'kitchen_to_table_time': 12,  # minutes
                    'checkout_time': 3  # minutes
                },
                'staff_productivity': {
                    'tables_per_server': 4,
                    'covers_per_cook': 25,
                    'dishes_per_dishwasher': 100
                }
            },
            'financial_metrics': {
                'food_cost_percentage': 0.32,
                'labor_cost_percentage': 0.28,
                'overhead_percentage': 0.25,
                'profit_margin': 0.15,
                'revenue_streams': {
                    'food_sales': 0.65,
                    'beverage_sales': 0.25,
                    'catering': 0.05,
                    'merchandise': 0.03,
                    'special_events': 0.02
                },
                'cost_breakdown': {
                    'ingredients': 0.32,
                    'labor': 0.28,
                    'rent': 0.08,
                    'utilities': 0.05,
                    'marketing': 0.04,
                    'maintenance': 0.03,
                    'insurance': 0.02,
                    'other': 0.18
                }
            },
            'sustainability_metrics': {
                'waste_reduction': {
                    'food_waste': 0.05,  # percentage
                    'packaging_waste': 0.03,
                    'water_usage': 0.02,
                    'energy_consumption': 0.04
                },
                'sourcing': {
                    'local_ingredients': 0.75,
                    'organic_ingredients': 0.45,
                    'sustainable_seafood': 0.90,
                    'fair_trade_products': 0.60
                },
                'energy_efficiency': {
                    'led_lighting': 0.95,
                    'energy_star_equipment': 0.85,
                    'smart_thermostats': 0.90,
                    'water_saving_devices': 0.80
                }
            }
        }
        
    def _initialize_certifications(self) -> Dict[str, Any]:
        """Initialize restaurant certifications and awards."""
        return {
            'food_safety': {
                'status': 'Certified',
                'last_inspection': '2024-01-15',
                'next_inspection': '2024-07-15',
                'certifications': [
                    'ServSafe',
                    'HACCP',
                    'ISO 22000',
                    'Allergen Management',
                    'Food Safety Modernization Act'
                ]
            },
            'sustainability': {
                'green_certification': 'Platinum',
                'waste_reduction': 'Gold',
                'energy_efficiency': 'Silver',
                'sustainable_sourcing': 'Gold',
                'water_conservation': 'Gold',
                'carbon_footprint_reduction': 'Silver'
            },
            'industry_awards': [
                'Best Fine Dining 2023',
                'Excellence in Service 2023',
                'Innovative Menu Design 2023',
                'Sustainable Restaurant of the Year 2023',
                'Wine Program Excellence 2023',
                'Chef of the Year 2023',
                'Best New Restaurant 2023',
                'Outstanding Hospitality 2023'
            ],
            'specialty_certifications': {
                'wine_program': 'Advanced Sommelier',
                'cheese_program': 'Certified Cheese Professional',
                'coffee_program': 'Specialty Coffee Association',
                'tea_program': 'Specialty Tea Association'
            }
        }
        
    def _initialize_restaurant_infrastructure(self) -> Dict[str, Any]:
        """Initialize restaurant infrastructure details."""
        return {
            'dining_areas': {
                'main_dining': {
                    'capacity': 100,
                    'tables': {
                        'two_seater': 20,
                        'four_seater': 15,
                        'six_seater': 5,
                        'private_booths': 10
                    },
                    'features': ['Ambient lighting', 'Live music area', 'Bar counter'],
                    'staff': {
                        'servers': 8,
                        'hosts': 2,
                        'bartenders': 3
                    }
                },
                'outdoor_patio': {
                    'capacity': 50,
                    'tables': {
                        'two_seater': 10,
                        'four_seater': 5,
                        'six_seater': 2
                    },
                    'features': ['Heating lamps', 'Garden view', 'Umbrella coverage'],
                    'staff': {
                        'servers': 4,
                        'hosts': 1
                    }
                },
                'private_rooms': {
                    'capacity': 40,
                    'rooms': {
                        'small': 2,
                        'medium': 1,
                        'large': 1
                    },
                    'features': ['AV equipment', 'Private bar', 'Catering kitchen'],
                    'staff': {
                        'dedicated_servers': 2,
                        'event_coordinator': 1
                    }
                }
            },
            'kitchen': {
                'stations': {
                    'hot_line': {
                        'equipment': ['6-burner range', '2 convection ovens', 'Salamander'],
                        'staff': ['Sous chef', 'Line cooks', 'Prep cooks'],
                        'capacity': '100 covers/hour'
                    },
                    'cold_line': {
                        'equipment': ['Prep tables', 'Refrigeration units', 'Salad station'],
                        'staff': ['Cold line chef', 'Prep cooks'],
                        'capacity': '50 covers/hour'
                    },
                    'pastry': {
                        'equipment': ['Pastry ovens', 'Mixers', 'Refrigeration'],
                        'staff': ['Pastry chef', 'Pastry cooks'],
                        'capacity': '30 desserts/hour'
                    }
                },
                'storage': {
                    'dry_storage': 200,  # sq ft
                    'refrigeration': 150,  # sq ft
                    'freezer': 100,  # sq ft
                    'wine_cellar': 300  # sq ft
                }
            },
            'equipment': {
                'cooking': {
                    'ranges': 4,
                    'ovens': 6,
                    'grills': 2,
                    'fryers': 2,
                    'steamers': 1
                },
                'refrigeration': {
                    'walk_in': 2,
                    'reach_in': 8,
                    'under_counter': 12
                },
                'preparation': {
                    'food_processors': 3,
                    'mixers': 2,
                    'slicers': 2
                }
            }
        }
        
    def _load_culinary_knowledge(self) -> Dict[str, Any]:
        """Load culinary knowledge base with dishes, ingredients, and techniques."""
        return {
            'cuisines': {
                'Contemporary American': {
                    'signature_dishes': [
                        'Pan-Seared Sea Bass',
                        'Wagyu Beef Tenderloin',
                        'Truffle Mac and Cheese',
                        'Lobster Bisque'
                    ],
                    'cooking_techniques': [
                        'Sous vide',
                        'Pan searing',
                        'Smoking',
                        'Braising'
                    ],
                    'key_ingredients': [
                        'Local produce',
                        'Sustainable seafood',
                        'Premium meats',
                        'Artisanal cheeses'
                    ],
                    'wine_pairings': [
                        'Chardonnay',
                        'Cabernet Sauvignon',
                        'Pinot Noir',
                        'Sauvignon Blanc'
                    ],
                    'dietary_accommodations': [
                        'Vegetarian',
                        'Vegan',
                        'Gluten-free',
                        'Dairy-free'
                    ]
                },
                'Mediterranean': {
                    'signature_dishes': [
                        'Grilled Octopus',
                        'Lamb Chops',
                        'Fattoush Salad',
                        'Hummus Trio'
                    ],
                    'cooking_techniques': [
                        'Grilling',
                        'Roasting',
                        'Marinating',
                        'Slow cooking'
                    ],
                    'key_ingredients': [
                        'Olive oil',
                        'Fresh herbs',
                        'Citrus',
                        'Legumes'
                    ],
                    'wine_pairings': [
                        'Rosé',
                        'Grenache',
                        'Tempranillo',
                        'Vermentino'
                    ],
                    'dietary_accommodations': [
                        'Vegetarian',
                        'Vegan',
                        'Gluten-free',
                        'Halal'
                    ]
                }
            },
            'ingredients': {
                'proteins': {
                    'beef': {
                        'cuts': ['Ribeye', 'Filet Mignon', 'Strip Steak', 'Wagyu'],
                        'grades': ['Prime', 'Choice', 'Select'],
                        'sourcing': ['Local farms', 'Premium suppliers'],
                        'storage': 'Refrigerated, 32-34°F'
                    },
                    'seafood': {
                        'types': ['Salmon', 'Sea Bass', 'Tuna', 'Lobster'],
                        'sourcing': ['Sustainable fisheries', 'Local suppliers'],
                        'storage': 'Refrigerated, 30-32°F',
                        'shelf_life': '2-3 days'
                    }
                },
                'produce': {
                    'vegetables': {
                        'categories': ['Root vegetables', 'Leafy greens', 'Nightshades'],
                        'sourcing': ['Local farms', 'Organic suppliers'],
                        'storage': 'Refrigerated, 35-38°F',
                        'shelf_life': '5-7 days'
                    },
                    'fruits': {
                        'categories': ['Citrus', 'Berries', 'Stone fruits'],
                        'sourcing': ['Local farms', 'Seasonal suppliers'],
                        'storage': 'Refrigerated, 35-38°F',
                        'shelf_life': '3-5 days'
                    }
                }
            },
            'techniques': {
                'basic': {
                    'knife_skills': ['Julienne', 'Brunoise', 'Chiffonade'],
                    'cooking_methods': ['Sautéing', 'Roasting', 'Grilling'],
                    'sauces': ['Reduction', 'Emulsification', 'Roux-based']
                },
                'advanced': {
                    'molecular': ['Spherification', 'Foams', 'Gels'],
                    'preservation': ['Pickling', 'Curing', 'Fermentation'],
                    'plating': ['Composition', 'Garnishing', 'Sauce work']
                }
            }
        }

    def _initialize_seasonal_items(self) -> Dict[str, Any]:
        """Initialize seasonal menu items and specials."""
        return {
            'spring': {
                'appetizers': [
                    'Spring Vegetable Medley',
                    'Fresh Asparagus Soup',
                    'Strawberry Spinach Salad',
                    'Herb-Infused Goat Cheese'
                ],
                'main_courses': [
                    'Spring Lamb with Mint Sauce',
                    'Pan-Seared Salmon with Spring Vegetables',
                    'Wild Mushroom Risotto',
                    'Spring Vegetable Pasta'
                ],
                'desserts': [
                    'Rhubarb Crumble',
                    'Lemon Lavender Cake',
                    'Strawberry Shortcake',
                    'Spring Berry Tart'
                ],
                'special_ingredients': [
                    'Fresh Asparagus',
                    'Spring Peas',
                    'Morel Mushrooms',
                    'Fresh Herbs',
                    'Spring Onions'
                ]
            },
            'summer': {
                'appetizers': [
                    'Chilled Gazpacho',
                    'Watermelon Feta Salad',
                    'Grilled Corn Salsa',
                    'Summer Caprese'
                ],
                'main_courses': [
                    'Grilled Seafood Platter',
                    'Summer Vegetable Paella',
                    'BBQ Ribs with Corn',
                    'Grilled Chicken with Summer Herbs'
                ],
                'desserts': [
                    'Peach Cobbler',
                    'Berry Pavlova',
                    'Ice Cream Sundae',
                    'Summer Fruit Tart'
                ],
                'special_ingredients': [
                    'Fresh Tomatoes',
                    'Sweet Corn',
                    'Summer Berries',
                    'Fresh Basil',
                    'Zucchini'
                ]
            },
            'fall': {
                'appetizers': [
                    'Butternut Squash Soup',
                    'Roasted Pumpkin Seeds',
                    'Apple Walnut Salad',
                    'Mushroom Pate'
                ],
                'main_courses': [
                    'Braised Short Ribs',
                    'Roasted Turkey with Gravy',
                    'Wild Mushroom Risotto',
                    'Pumpkin Ravioli'
                ],
                'desserts': [
                    'Apple Pie',
                    'Pumpkin Cheesecake',
                    'Pecan Tart',
                    'Spiced Pear Crisp'
                ],
                'special_ingredients': [
                    'Pumpkin',
                    'Apples',
                    'Wild Mushrooms',
                    'Sweet Potatoes',
                    'Cranberries'
                ]
            },
            'winter': {
                'appetizers': [
                    'Creamy Mushroom Soup',
                    'Winter Root Vegetable Salad',
                    'Baked Brie with Cranberries',
                    'Roasted Chestnut Soup'
                ],
                'main_courses': [
                    'Beef Wellington',
                    'Roasted Duck with Orange Sauce',
                    'Winter Vegetable Stew',
                    'Braised Lamb Shank'
                ],
                'desserts': [
                    'Chocolate Yule Log',
                    'Winter Spice Cake',
                    'Cranberry Bread Pudding',
                    'Gingerbread Souffle'
                ],
                'special_ingredients': [
                    'Winter Squash',
                    'Brussels Sprouts',
                    'Chestnuts',
                    'Citrus Fruits',
                    'Root Vegetables'
                ]
            },
            'holiday_specials': {
                'christmas': [
                    'Roast Turkey with All Trimmings',
                    'Christmas Pudding',
                    'Mince Pies',
                    'Mulled Wine'
                ],
                'new_year': [
                    'Champagne Brunch Special',
                    'Lobster Thermidor',
                    'Chocolate Souffle',
                    'Sparkling Wine Pairing'
                ],
                'valentines': [
                    'Chocolate-Covered Strawberries',
                    'Romantic Dinner for Two',
                    'Champagne Truffles',
                    'Love Potion Cocktail'
                ],
                'thanksgiving': [
                    'Traditional Turkey Dinner',
                    'Pumpkin Pie',
                    'Cranberry Sauce',
                    'Sweet Potato Casserole'
                ]
            },
            'daily_specials': {
                'monday': 'Chef\'s Pasta Special',
                'tuesday': 'Taco Tuesday',
                'wednesday': 'Wine Wednesday',
                'thursday': 'Thirsty Thursday',
                'friday': 'Fish Friday',
                'saturday': 'Weekend Brunch',
                'sunday': 'Sunday Roast'
            }
        }

    def generate_customer_demographics(self, num_customers=1000):
        """Generate customer demographic data."""
        customers = []
        for _ in range(num_customers):
            customer = {
                'customer_id': self.fake.uuid4(),
                'name': self.fake.name(),
                'age': self._generate_realistic_age(),
                'email': self.fake.email(),
                'phone': self.fake.phone_number(),
                'address': self.fake.address(),
                'preferences': self._generate_dining_preferences(),
                'dietary_restrictions': self._generate_dietary_restrictions(),
                'loyalty_status': self._generate_loyalty_status(),
                'visit_frequency': self._generate_visit_frequency(),
                'average_spend': self._generate_average_spend(),
                'preferred_dining_times': self._generate_preferred_dining_times(),
                'preferred_cuisines': self._generate_preferred_cuisines(),
                'special_occasions': self._generate_special_occasions()
            }
            customers.append(customer)
        return pd.DataFrame(customers)

    def _generate_realistic_age(self) -> int:
        """Generate a realistic age distribution."""
        age_groups = {
            '18-24': 0.15,
            '25-34': 0.25,
            '35-44': 0.20,
            '45-54': 0.15,
            '55-64': 0.15,
            '65+': 0.10
        }
        age_group = random.choices(list(age_groups.keys()), weights=list(age_groups.values()))[0]
        
        if age_group == '65+':
            return random.randint(65, 85)  # Cap at 85 for realistic distribution
        else:
            min_age, max_age = map(int, age_group.split('-'))
            return random.randint(min_age, max_age)

    def _generate_dining_preferences(self) -> Dict[str, Any]:
        """Generate customer dining preferences."""
        return {
            'seating_preference': random.choice(['indoor', 'outdoor', 'bar', 'private_room']),
            'noise_level': random.choice(['quiet', 'moderate', 'lively']),
            'lighting': random.choice(['dim', 'moderate', 'bright']),
            'music_preference': random.choice(['none', 'background', 'live']),
            'table_size': random.choice(['2-top', '4-top', '6-top', '8-top']),
            'special_requests': random.sample([
                'window seat',
                'booth',
                'quiet corner',
                'near kitchen',
                'away from kitchen'
            ], random.randint(0, 2))
        }

    def _generate_dietary_restrictions(self) -> List[str]:
        """Generate dietary restrictions and preferences."""
        restrictions = []
        if random.random() < 0.3:  # 30% chance of having dietary restrictions
            possible_restrictions = [
                'vegetarian',
                'vegan',
                'gluten-free',
                'dairy-free',
                'nut-free',
                'shellfish-free',
                'halal',
                'kosher'
            ]
            restrictions = random.sample(possible_restrictions, random.randint(1, 3))
        return restrictions

    def _generate_loyalty_status(self) -> str:
        """Generate customer loyalty status."""
        statuses = {
            'bronze': 0.4,
            'silver': 0.3,
            'gold': 0.2,
            'platinum': 0.1
        }
        return random.choices(list(statuses.keys()), weights=list(statuses.values()))[0]

    def _generate_visit_frequency(self) -> Dict[str, Any]:
        """Generate customer visit frequency data."""
        return {
            'visits_per_month': random.randint(1, 8),
            'last_visit': self.fake.date_time_between(start_date='-365d', end_date='now'),
            'average_visit_duration': random.randint(45, 180),  # minutes
            'preferred_visit_days': random.sample(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                                               random.randint(1, 7)),
            'visit_pattern': random.choice(['regular', 'occasional', 'seasonal', 'special_occasions'])
        }

    def _generate_average_spend(self) -> Dict[str, float]:
        """Generate customer average spending data."""
        return {
            'per_visit': round(random.uniform(30.0, 200.0), 2),
            'per_month': round(random.uniform(100.0, 1000.0), 2),
            'per_year': round(random.uniform(1000.0, 10000.0), 2)
        }

    def _generate_preferred_dining_times(self) -> List[str]:
        """Generate preferred dining times."""
        times = ['breakfast', 'lunch', 'dinner', 'late_night']
        return random.sample(times, random.randint(1, 4))

    def _generate_preferred_cuisines(self) -> List[str]:
        """Generate preferred cuisines."""
        cuisines = list(self.culinary_knowledge['cuisines'].keys())
        return random.sample(cuisines, random.randint(1, len(cuisines)))

    def _generate_special_occasions(self) -> List[Dict[str, Any]]:
        """Generate special occasion data."""
        occasions = []
        if random.random() < 0.4:  # 40% chance of having special occasions
            possible_occasions = [
                'birthday',
                'anniversary',
                'business_meeting',
                'family_gathering',
                'date_night',
                'celebration'
            ]
            num_occasions = random.randint(1, 3)
            for _ in range(num_occasions):
                occasion = {
                    'type': random.choice(possible_occasions),
                    'frequency': random.choice(['monthly', 'quarterly', 'annually', 'special']),
                    'preferred_seating': random.choice(['private_room', 'main_dining', 'outdoor']),
                    'average_group_size': random.randint(2, 20)
                }
                occasions.append(occasion)
        return occasions

    def generate_reservations(self, customer_df, num_reservations=2000):
        """Generate restaurant reservation data."""
        reservations = []
        for _ in range(num_reservations):
            customer = customer_df.sample(n=1).iloc[0]
            reservation_time = self._generate_reservation_time()
            party_size = self._generate_party_size()
            reservation = {
                'reservation_id': self.fake.uuid4(),
                'customer_id': customer['customer_id'],
                'reservation_time': reservation_time,
                'party_size': party_size,
                'seating_area': self._generate_seating_area(party_size),
                'special_requests': self._generate_special_requests(customer),
                'status': self._generate_reservation_status(reservation_time),
                'duration': self._generate_reservation_duration(party_size),
                'occasion': self._generate_reservation_occasion(),
                'preferred_server': self._generate_preferred_server(),
                'table_preference': self._generate_table_preference(party_size)
            }
            reservations.append(reservation)
        return pd.DataFrame(reservations)

    def _generate_reservation_time(self) -> datetime:
        """Generate a realistic reservation time."""
        now = datetime.now()
        future_date = now + timedelta(days=random.randint(0, 30))
        hour = random.randint(11, 21)  # Between 11 AM and 9 PM
        minute = random.choice([0, 15, 30, 45])
        return future_date.replace(hour=hour, minute=minute)

    def _generate_party_size(self) -> int:
        """Generate a realistic party size."""
        party_sizes = {
            1: 0.05,  # 5% solo diners
            2: 0.40,  # 40% couples
            3: 0.15,  # 15% three people
            4: 0.25,  # 25% four people
            5: 0.05,  # 5% five people
            6: 0.05,  # 5% six people
            7: 0.02,  # 2% seven people
            8: 0.02,  # 2% eight people
            9: 0.005, # 0.5% nine people
            10: 0.005 # 0.5% ten people
        }
        return random.choices(list(party_sizes.keys()), weights=list(party_sizes.values()))[0]

    def _generate_seating_area(self, party_size: int) -> str:
        """Generate appropriate seating area based on party size."""
        if party_size <= 2:
            return random.choice(['main_dining', 'bar', 'outdoor_patio'])
        elif party_size <= 4:
            return random.choice(['main_dining', 'outdoor_patio'])
        elif party_size <= 6:
            return 'main_dining'
        else:
            return 'private_room'

    def _generate_special_requests(self, customer: pd.Series) -> List[str]:
        """Generate special requests for a reservation."""
        requests = []
        if random.random() < 0.3:  # 30% chance of having special requests
            possible_requests = [
                'high chair',
                'wheelchair accessible',
                'quiet table',
                'window seat',
                'booth',
                'birthday celebration',
                'anniversary celebration',
                'allergy considerations',
                'special dietary requirements'
            ]
            requests = random.sample(possible_requests, random.randint(1, 3))
        return requests

    def _generate_customer_special_requests(self) -> Dict[str, Any]:
        """Generate customer special requests data."""
        return {
            'dietary_restrictions': random.sample([
                'vegetarian',
                'vegan',
                'gluten_free',
                'dairy_free',
                'nut_free',
                'shellfish_free',
                'halal',
                'kosher'
            ], random.randint(0, 3)),
            'allergies': random.sample([
                'nuts',
                'shellfish',
                'dairy',
                'eggs',
                'soy',
                'wheat'
            ], random.randint(0, 2)),
            'seating_preferences': random.sample([
                'window',
                'quiet',
                'booth',
                'bar',
                'outdoor'
            ], random.randint(0, 2)),
            'special_occasions': random.sample([
                'birthday',
                'anniversary',
                'business_meeting',
                'celebration'
            ], random.randint(0, 1))
        }

    def _generate_reservation_status(self, reservation_time: datetime) -> str:
        """Generate reservation status."""
        now = datetime.now()
        if reservation_time < now:
            return random.choice(['completed', 'no_show', 'cancelled'])
        else:
            return random.choice(['confirmed', 'pending', 'cancelled'])

    def _generate_reservation_duration(self, party_size: int) -> int:
        """Generate expected reservation duration in minutes."""
        base_duration = 90  # Base duration for 2 people
        return base_duration + (party_size - 2) * 15  # Add 15 minutes per additional person

    def _generate_reservation_occasion(self) -> str:
        """Generate reservation occasion."""
        occasions = {
            'regular_dining': 0.6,
            'birthday': 0.1,
            'anniversary': 0.1,
            'business_meeting': 0.1,
            'special_celebration': 0.1
        }
        return random.choices(list(occasions.keys()), weights=list(occasions.values()))[0]

    def _generate_preferred_server(self) -> str:
        """Generate preferred server assignment."""
        servers = [
            'John Smith',
            'Sarah Johnson',
            'Michael Brown',
            'Emily Davis',
            'David Wilson',
            'Lisa Anderson',
            'James Taylor',
            'Jennifer Martinez'
        ]
        return random.choice(servers)

    def _generate_table_preference(self, party_size: int) -> str:
        """Generate table preference based on party size."""
        if party_size <= 2:
            return random.choice(['2-top', 'booth', 'bar'])
        elif party_size <= 4:
            return random.choice(['4-top', 'booth'])
        elif party_size <= 6:
            return '6-top'
        else:
            return f'{party_size}-top'

    def generate_orders(self, customer_df, num_orders=5000):
        """Generate restaurant order data."""
        orders = []
        for _ in range(num_orders):
            customer = customer_df.sample(n=1).iloc[0]
            order_time = self._generate_order_time()
            order = {
                'order_id': self.fake.uuid4(),
                'customer_id': customer['customer_id'],
                'order_time': order_time,
                'items': self._generate_order_items(),
                'total_amount': 0.0,  # Will be calculated after items
                'payment_method': self._generate_payment_method(),
                'service_type': self._generate_service_type(),
                'special_instructions': self._generate_special_instructions(),
                'server': self._generate_server(),
                'table_number': self._generate_table_number(),
                'order_status': self._generate_order_status(),
                'preparation_time': self._generate_preparation_time(),
                'delivery_time': self._generate_delivery_time()
            }
            # Calculate total amount
            order['total_amount'] = sum(item['price'] * item['quantity'] for item in order['items'])
            orders.append(order)
        return pd.DataFrame(orders)

    def _generate_order_time(self) -> datetime:
        """Generate a realistic order time."""
        now = datetime.now()
        past_date = now - timedelta(days=random.randint(0, 30))
        hour = random.randint(11, 21)  # Between 11 AM and 9 PM
        minute = random.randint(0, 59)
        return past_date.replace(hour=hour, minute=minute)

    def _generate_order_items(self) -> List[Dict[str, Any]]:
        """Generate order items with realistic quantities and prices."""
        items = []
        num_items = random.randint(1, 5)
        menu_items = self._get_menu_items()
        
        for _ in range(num_items):
            item = random.choice(menu_items)
            quantity = random.randint(1, 3)
            items.append({
                'item_id': item['id'],
                'name': item['name'],
                'category': item['category'],
                'price': item['price'],
                'quantity': quantity,
                'special_instructions': self._generate_item_special_instructions()
            })
        return items

    def _get_menu_items(self) -> List[Dict[str, Any]]:
        """Get menu items with prices and categories."""
        return [
            {'id': 1, 'name': 'Wagyu Beef Tenderloin', 'category': 'main', 'price': 65.00},
            {'id': 2, 'name': 'Pan-Seared Sea Bass', 'category': 'main', 'price': 45.00},
            {'id': 3, 'name': 'Truffle Mac and Cheese', 'category': 'side', 'price': 18.00},
            {'id': 4, 'name': 'Lobster Bisque', 'category': 'appetizer', 'price': 16.00},
            {'id': 5, 'name': 'Grilled Octopus', 'category': 'appetizer', 'price': 22.00},
            {'id': 6, 'name': 'Lamb Chops', 'category': 'main', 'price': 48.00},
            {'id': 7, 'name': 'Fattoush Salad', 'category': 'appetizer', 'price': 14.00},
            {'id': 8, 'name': 'Hummus Trio', 'category': 'appetizer', 'price': 12.00},
            {'id': 9, 'name': 'Chocolate Soufflé', 'category': 'dessert', 'price': 12.00},
            {'id': 10, 'name': 'Crème Brûlée', 'category': 'dessert', 'price': 10.00}
        ]

    def _generate_payment_method(self) -> str:
        """Generate payment method."""
        methods = {
            'credit_card': 0.6,
            'debit_card': 0.2,
            'cash': 0.1,
            'mobile_payment': 0.1
        }
        return random.choices(list(methods.keys()), weights=list(methods.values()))[0]

    def _generate_service_type(self) -> str:
        """Generate service type."""
        types = {
            'dine_in': 0.7,
            'takeout': 0.2,
            'delivery': 0.1
        }
        return random.choices(list(types.keys()), weights=list(types.values()))[0]

    def _generate_special_instructions(self) -> str:
        """Generate special instructions for the order."""
        instructions = []
        if random.random() < 0.3:  # 30% chance of special instructions
            possible_instructions = [
                'allergies',
                'spice_level',
                'cooking_preference',
                'dietary_restrictions',
                'special_requests'
            ]
            instructions = random.sample(possible_instructions, random.randint(1, 2))
        return ', '.join(instructions) if instructions else ''

    def _generate_server(self) -> str:
        """Generate server name."""
        servers = [
            'John Smith',
            'Sarah Johnson',
            'Michael Brown',
            'Emily Davis',
            'David Wilson',
            'Lisa Anderson',
            'James Taylor',
            'Jennifer Martinez'
        ]
        return random.choice(servers)

    def _generate_table_number(self) -> int:
        """Generate table number."""
        return random.randint(1, 50)

    def _generate_order_status(self) -> str:
        """Generate order status."""
        statuses = {
            'completed': 0.8,
            'cancelled': 0.1,
            'refunded': 0.05,
            'in_progress': 0.05
        }
        return random.choices(list(statuses.keys()), weights=list(statuses.values()))[0]

    def _generate_preparation_time(self) -> int:
        """Generate food preparation time in minutes."""
        return random.randint(15, 45)

    def _generate_delivery_time(self) -> int:
        """Generate delivery time in minutes."""
        return random.randint(30, 90)

    def _generate_item_special_instructions(self) -> str:
        """Generate special instructions for individual items."""
        instructions = []
        if random.random() < 0.2:  # 20% chance of special instructions
            possible_instructions = [
                'no onions',
                'extra spicy',
                'well done',
                'medium rare',
                'no dairy',
                'gluten-free',
                'extra sauce',
                'light sauce'
            ]
            instructions = random.sample(possible_instructions, random.randint(1, 2))
        return ', '.join(instructions) if instructions else ''

    def generate_dataset(self):
        """Generate complete restaurant dataset."""
        # Generate customer demographics
        customer_df = self.generate_customer_demographics(num_customers=1000)
        
        # Generate reservations
        reservation_df = self.generate_reservations(customer_df, num_reservations=2000)
        
        # Generate orders
        order_df = self.generate_orders(customer_df, num_orders=5000)
        
        # Combine all data into a single dataset
        dataset = {
            'customers': customer_df,
            'reservations': reservation_df,
            'orders': order_df,
            'restaurant_info': {
                'name': self.restaurant_name,
                'owner': self.owner,
                'infrastructure': self.infrastructure,
                'quality_metrics': self.quality_metrics,
                'certifications': self.certifications
            }
        }
        
        return dataset 

    def generate_customer_behavior(self, customer_df, num_behaviors=1000):
        """Generate customer behavior data."""
        behaviors = []
        for _ in range(num_behaviors):
            behavior = {
                'customer_id': random.choice(customer_df['customer_id']),
                'visit_pattern': self._generate_visit_pattern(),
                'ordering_behavior': self._generate_ordering_behavior(),
                'payment_behavior': self._generate_payment_behavior(),
                'feedback_behavior': self._generate_feedback_behavior(),
                'special_requests': self._generate_customer_special_requests(),
                'social_behavior': self._generate_social_behavior(),
                'loyalty_behavior': self._generate_loyalty_behavior(),
                'seasonal_preferences': self._generate_seasonal_preferences(),
                'dining_companions': self._generate_dining_companions(),
                'communication_preferences': self._generate_communication_preferences()
            }
            behaviors.append(behavior)
        return pd.DataFrame(behaviors)

    def _generate_visit_pattern(self) -> Dict[str, Any]:
        """Generate customer visit pattern data."""
        return {
            'frequency': {
                'weekly': random.randint(0, 3),
                'monthly': random.randint(1, 8),
                'annual': random.randint(12, 52)
            },
            'preferred_days': random.sample(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                                         random.randint(1, 7)),
            'preferred_times': {
                'breakfast': random.random() < 0.2,
                'lunch': random.random() < 0.4,
                'dinner': random.random() < 0.8,
                'late_night': random.random() < 0.1
            },
            'average_stay_duration': random.randint(60, 180),  # minutes
            'seasonal_variation': {
                'summer': random.uniform(0.8, 1.2),
                'winter': random.uniform(0.8, 1.2),
                'spring': random.uniform(0.8, 1.2),
                'fall': random.uniform(0.8, 1.2)
            }
        }

    def _generate_ordering_behavior(self) -> Dict[str, Any]:
        """Generate customer ordering behavior data."""
        return {
            'course_preference': {
                'appetizer': random.random() < 0.7,
                'main_course': random.random() < 0.9,
                'dessert': random.random() < 0.5,
                'beverages': random.random() < 0.8
            },
            'portion_preference': random.choice(['small', 'regular', 'large']),
            'spice_preference': random.choice(['mild', 'medium', 'spicy']),
            'cooking_preference': {
                'rare': random.random() < 0.2,
                'medium_rare': random.random() < 0.3,
                'medium': random.random() < 0.3,
                'well_done': random.random() < 0.2
            },
            'modification_frequency': random.uniform(0, 1),
            'sharing_behavior': random.random() < 0.4,
            'wine_pairing': random.random() < 0.6
        }

    def _generate_payment_behavior(self) -> Dict[str, Any]:
        """Generate customer payment behavior data."""
        return {
            'payment_method': {
                'credit_card': random.random() < 0.6,
                'debit_card': random.random() < 0.2,
                'cash': random.random() < 0.1,
                'mobile_payment': random.random() < 0.1
            },
            'tip_percentage': random.uniform(0.15, 0.25),
            'split_bill_frequency': random.uniform(0, 1),
            'loyalty_points_usage': random.uniform(0, 1),
            'special_offers_usage': random.uniform(0, 1),
            'payment_timing': random.choice(['immediate', 'end_of_meal', 'pre_payment'])
        }

    def _generate_feedback_behavior(self) -> Dict[str, Any]:
        """Generate customer feedback behavior data."""
        return {
            'review_frequency': random.uniform(0, 1),
            'review_platforms': random.sample(['google', 'yelp', 'tripadvisor', 'social_media'], 
                                           random.randint(0, 4)),
            'feedback_type': {
                'positive': random.uniform(0.6, 0.9),
                'constructive': random.uniform(0.1, 0.3),
                'negative': random.uniform(0, 0.1)
            },
            'response_to_feedback': random.random() < 0.3,
            'social_sharing': random.random() < 0.4,
            'photo_sharing': random.random() < 0.3
        }

    def _generate_social_behavior(self) -> Dict[str, Any]:
        """Generate customer social behavior data."""
        return {
            'group_size': {
                'solo': random.random() < 0.2,
                'couple': random.random() < 0.4,
                'small_group': random.random() < 0.3,
                'large_group': random.random() < 0.1
            },
            'social_interaction': {
                'staff_interaction': random.uniform(0, 1),
                'other_customers': random.uniform(0, 1),
                'social_media_checkin': random.random() < 0.3
            },
            'dining_companions': {
                'family': random.random() < 0.3,
                'friends': random.random() < 0.4,
                'colleagues': random.random() < 0.2,
                'dates': random.random() < 0.2
            },
            'social_sharing': {
                'photos': random.random() < 0.4,
                'reviews': random.random() < 0.3,
                'recommendations': random.random() < 0.5
            }
        }

    def _generate_loyalty_behavior(self) -> Dict[str, Any]:
        """Generate customer loyalty behavior data."""
        return {
            'loyalty_program_engagement': random.uniform(0, 1),
            'points_accumulation_rate': random.uniform(0, 1),
            'points_redemption_rate': random.uniform(0, 1),
            'referral_behavior': random.uniform(0, 1),
            'special_offers_usage': random.uniform(0, 1),
            'loyalty_tier': random.choice(['bronze', 'silver', 'gold', 'platinum']),
            'membership_duration': random.randint(0, 60),  # months
            'engagement_frequency': random.uniform(0, 1)
        }

    def _generate_seasonal_preferences(self) -> Dict[str, Any]:
        """Generate customer seasonal preferences data."""
        return {
            'seasonal_visits': {
                'spring': random.uniform(0.8, 1.2),
                'summer': random.uniform(0.8, 1.2),
                'fall': random.uniform(0.8, 1.2),
                'winter': random.uniform(0.8, 1.2)
            },
            'seasonal_menu_preferences': {
                'spring': random.sample(self.seasonal_items['spring']['main_courses'], 
                                     random.randint(1, 3)),
                'summer': random.sample(self.seasonal_items['summer']['main_courses'], 
                                     random.randint(1, 3)),
                'fall': random.sample(self.seasonal_items['fall']['main_courses'], 
                                   random.randint(1, 3)),
                'winter': random.sample(self.seasonal_items['winter']['main_courses'], 
                                     random.randint(1, 3))
            },
            'special_occasions': {
                'holidays': random.random() < 0.3,
                'birthdays': random.random() < 0.4,
                'anniversaries': random.random() < 0.3,
                'business_events': random.random() < 0.2
            }
        }

    def _generate_dining_companions(self) -> Dict[str, Any]:
        """Generate customer dining companions data."""
        return {
            'regular_companions': random.randint(0, 5),
            'companion_types': {
                'family': random.random() < 0.4,
                'friends': random.random() < 0.6,
                'colleagues': random.random() < 0.3,
                'dates': random.random() < 0.2
            },
            'group_dynamics': {
                'decision_maker': random.random() < 0.3,
                'influencer': random.random() < 0.4,
                'follower': random.random() < 0.3
            },
            'social_interaction': {
                'high': random.random() < 0.3,
                'medium': random.random() < 0.5,
                'low': random.random() < 0.2
            }
        }

    def _generate_communication_preferences(self) -> Dict[str, Any]:
        """Generate customer communication preferences data."""
        return {
            'preferred_channels': {
                'email': random.random() < 0.7,
                'sms': random.random() < 0.5,
                'phone': random.random() < 0.3,
                'social_media': random.random() < 0.4
            },
            'communication_frequency': {
                'promotions': random.choice(['daily', 'weekly', 'monthly', 'never']),
                'newsletters': random.choice(['weekly', 'monthly', 'quarterly', 'never']),
                'special_offers': random.choice(['immediate', 'weekly', 'monthly', 'never'])
            },
            'response_rate': random.uniform(0, 1),
            'preferred_language': random.choice(['english', 'spanish', 'french', 'mandarin']),
            'accessibility_needs': {
                'hearing': random.random() < 0.1,
                'visual': random.random() < 0.1,
                'mobility': random.random() < 0.1
            }
        }

    def generate_restaurant_operations(self, num_operations=1000):
        """Generate detailed restaurant operations data."""
        operations = []
        for _ in range(num_operations):
            operation = {
                'kitchen_operations': self._generate_kitchen_operations(),
                'service_operations': self._generate_service_operations(),
                'inventory_operations': self._generate_inventory_operations(),
                'staff_operations': self._generate_staff_operations(),
                'maintenance_operations': self._generate_maintenance_operations(),
                'quality_control': self._generate_quality_control(),
                'financial_operations': self._generate_financial_operations(),
                'marketing_operations': self._generate_marketing_operations(),
                'safety_operations': self._generate_safety_operations(),
                'sustainability_operations': self._generate_sustainability_operations()
            }
            operations.append(operation)
        return pd.DataFrame(operations)

    def _generate_kitchen_operations(self) -> Dict[str, Any]:
        """Generate kitchen operations data."""
        return {
            'prep_schedule': {
                'morning_prep': random.randint(2, 4),  # hours
                'afternoon_prep': random.randint(1, 3),  # hours
                'evening_prep': random.randint(1, 2)  # hours
            },
            'cooking_stations': {
                'hot_line': {
                    'efficiency': random.uniform(0.8, 1.0),
                    'downtime': random.uniform(0, 0.1),
                    'quality_score': random.uniform(0.8, 1.0)
                },
                'cold_line': {
                    'efficiency': random.uniform(0.8, 1.0),
                    'downtime': random.uniform(0, 0.1),
                    'quality_score': random.uniform(0.8, 1.0)
                },
                'pastry': {
                    'efficiency': random.uniform(0.8, 1.0),
                    'downtime': random.uniform(0, 0.1),
                    'quality_score': random.uniform(0.8, 1.0)
                }
            },
            'equipment_usage': {
                'ovens': random.uniform(0.6, 0.9),
                'ranges': random.uniform(0.7, 0.95),
                'refrigeration': random.uniform(0.8, 1.0),
                'prep_equipment': random.uniform(0.7, 0.9)
            },
            'waste_management': {
                'food_waste': random.uniform(0.02, 0.08),
                'packaging_waste': random.uniform(0.01, 0.05),
                'recycling_rate': random.uniform(0.7, 0.9)
            }
        }

    def _generate_service_operations(self) -> Dict[str, Any]:
        """Generate service operations data."""
        return {
            'table_management': {
                'turnover_rate': random.uniform(2.0, 3.0),
                'average_occupancy': random.uniform(0.7, 0.9),
                'wait_times': random.uniform(5, 15)  # minutes
            },
            'service_quality': {
                'staff_knowledge': random.uniform(0.8, 1.0),
                'response_time': random.uniform(0.8, 1.0),
                'customer_interaction': random.uniform(0.8, 1.0)
            },
            'reservation_management': {
                'booking_rate': random.uniform(0.7, 0.9),
                'no_show_rate': random.uniform(0.05, 0.15),
                'cancellation_rate': random.uniform(0.1, 0.2)
            },
            'special_requests': {
                'fulfillment_rate': random.uniform(0.9, 1.0),
                'response_time': random.uniform(0.8, 1.0),
                'customer_satisfaction': random.uniform(0.8, 1.0)
            }
        }

    def _generate_inventory_operations(self) -> Dict[str, Any]:
        """Generate inventory operations data."""
        return {
            'stock_levels': {
                'perishables': random.uniform(0.7, 0.9),
                'non_perishables': random.uniform(0.6, 0.8),
                'beverages': random.uniform(0.7, 0.9)
            },
            'ordering': {
                'frequency': random.choice(['daily', 'weekly', 'bi-weekly']),
                'accuracy': random.uniform(0.8, 1.0),
                'lead_time': random.uniform(1, 3)  # days
            },
            'waste_management': {
                'spoilage_rate': random.uniform(0.01, 0.05),
                'inventory_turnover': random.uniform(3, 7),  # times per week
                'recycling_rate': random.uniform(0.7, 0.9)
            },
            'supplier_performance': {
                'delivery_reliability': random.uniform(0.8, 1.0),
                'quality_consistency': random.uniform(0.8, 1.0),
                'price_stability': random.uniform(0.7, 0.9)
            }
        }

    def _generate_staff_operations(self) -> Dict[str, Any]:
        """Generate staff operations data."""
        return {
            'scheduling': {
                'coverage_rate': random.uniform(0.8, 1.0),
                'overtime_rate': random.uniform(0.05, 0.15),
                'staff_utilization': random.uniform(0.7, 0.9)
            },
            'training': {
                'completion_rate': random.uniform(0.8, 1.0),
                'certification_rate': random.uniform(0.7, 0.9),
                'skill_development': random.uniform(0.7, 0.9)
            },
            'performance': {
                'productivity': random.uniform(0.8, 1.0),
                'quality': random.uniform(0.8, 1.0),
                'customer_satisfaction': random.uniform(0.8, 1.0)
            },
            'retention': {
                'turnover_rate': random.uniform(0.1, 0.3),
                'satisfaction_rate': random.uniform(0.7, 0.9),
                'promotion_rate': random.uniform(0.1, 0.3)
            }
        }

    def _generate_maintenance_operations(self) -> Dict[str, Any]:
        """Generate maintenance operations data."""
        return {
            'preventive_maintenance': {
                'schedule_compliance': random.uniform(0.8, 1.0),
                'equipment_uptime': random.uniform(0.9, 1.0),
                'maintenance_cost': random.uniform(0.02, 0.05)  # percentage of revenue
            },
            'repairs': {
                'response_time': random.uniform(0.7, 0.9),
                'resolution_rate': random.uniform(0.8, 1.0),
                'cost_efficiency': random.uniform(0.7, 0.9)
            },
            'facility_management': {
                'cleanliness': random.uniform(0.8, 1.0),
                'safety_compliance': random.uniform(0.9, 1.0),
                'energy_efficiency': random.uniform(0.7, 0.9)
            },
            'equipment_lifecycle': {
                'replacement_schedule': random.uniform(0.8, 1.0),
                'upgrade_rate': random.uniform(0.1, 0.3),
                'depreciation_rate': random.uniform(0.1, 0.2)
            }
        }

    def _generate_quality_control(self) -> Dict[str, Any]:
        """Generate quality control data."""
        return {
            'food_quality': {
                'temperature_control': random.uniform(0.9, 1.0),
                'presentation': random.uniform(0.8, 1.0),
                'taste_consistency': random.uniform(0.8, 1.0)
            },
            'service_quality': {
                'response_time': random.uniform(0.8, 1.0),
                'accuracy': random.uniform(0.8, 1.0),
                'professionalism': random.uniform(0.8, 1.0)
            },
            'facility_quality': {
                'cleanliness': random.uniform(0.8, 1.0),
                'maintenance': random.uniform(0.8, 1.0),
                'ambiance': random.uniform(0.8, 1.0)
            },
            'compliance': {
                'health_regulations': random.uniform(0.9, 1.0),
                'safety_standards': random.uniform(0.9, 1.0),
                'operational_procedures': random.uniform(0.8, 1.0)
            }
        }

    def _generate_financial_operations(self) -> Dict[str, Any]:
        """Generate financial operations data."""
        return {
            'revenue_management': {
                'average_check': random.uniform(40, 60),
                'table_turnover': random.uniform(2.0, 3.0),
                'upselling_rate': random.uniform(0.2, 0.4)
            },
            'cost_control': {
                'food_cost': random.uniform(0.28, 0.35),
                'labor_cost': random.uniform(0.25, 0.32),
                'overhead_cost': random.uniform(0.20, 0.28)
            },
            'profitability': {
                'gross_margin': random.uniform(0.65, 0.75),
                'net_margin': random.uniform(0.10, 0.20),
                'return_on_investment': random.uniform(0.15, 0.25)
            },
            'cash_flow': {
                'daily_revenue': random.uniform(5000, 15000),
                'payment_processing': random.uniform(0.95, 1.0),
                'inventory_investment': random.uniform(0.15, 0.25)
            }
        }

    def _generate_marketing_operations(self) -> Dict[str, Any]:
        """Generate marketing operations data."""
        return {
            'campaign_performance': {
                'email_marketing': random.uniform(0.2, 0.4),
                'social_media': random.uniform(0.3, 0.5),
                'loyalty_program': random.uniform(0.4, 0.6)
            },
            'customer_acquisition': {
                'new_customers': random.uniform(0.1, 0.3),
                'conversion_rate': random.uniform(0.2, 0.4),
                'acquisition_cost': random.uniform(20, 40)
            },
            'customer_retention': {
                'repeat_rate': random.uniform(0.4, 0.6),
                'loyalty_rate': random.uniform(0.3, 0.5),
                'churn_rate': random.uniform(0.1, 0.3)
            },
            'brand_metrics': {
                'awareness': random.uniform(0.6, 0.8),
                'reputation': random.uniform(0.7, 0.9),
                'social_media_engagement': random.uniform(0.3, 0.5)
            }
        }

    def _generate_safety_operations(self) -> Dict[str, Any]:
        """Generate safety operations data."""
        return {
            'food_safety': {
                'temperature_control': random.uniform(0.9, 1.0),
                'cross_contamination': random.uniform(0.9, 1.0),
                'allergen_management': random.uniform(0.9, 1.0)
            },
            'workplace_safety': {
                'accident_rate': random.uniform(0, 0.1),
                'safety_training': random.uniform(0.8, 1.0),
                'compliance_rate': random.uniform(0.9, 1.0)
            },
            'emergency_preparedness': {
                'response_time': random.uniform(0.8, 1.0),
                'drill_frequency': random.uniform(0.7, 0.9),
                'equipment_maintenance': random.uniform(0.8, 1.0)
            },
            'security_measures': {
                'access_control': random.uniform(0.8, 1.0),
                'surveillance': random.uniform(0.8, 1.0),
                'incident_response': random.uniform(0.8, 1.0)
            }
        }

    def _generate_sustainability_operations(self) -> Dict[str, Any]:
        """Generate sustainability operations data."""
        return {
            'waste_management': {
                'recycling_rate': random.uniform(0.7, 0.9),
                'composting_rate': random.uniform(0.5, 0.7),
                'waste_reduction': random.uniform(0.6, 0.8)
            },
            'energy_management': {
                'energy_efficiency': random.uniform(0.7, 0.9),
                'renewable_energy': random.uniform(0.3, 0.5),
                'carbon_footprint': random.uniform(0.6, 0.8)
            },
            'water_management': {
                'water_conservation': random.uniform(0.7, 0.9),
                'water_recycling': random.uniform(0.5, 0.7),
                'water_efficiency': random.uniform(0.7, 0.9)
            },
            'sustainable_sourcing': {
                'local_ingredients': random.uniform(0.6, 0.8),
                'organic_ingredients': random.uniform(0.4, 0.6),
                'fair_trade_products': random.uniform(0.5, 0.7)
            }
        }

    def generate_menu_data(self, num_items=100):
        """Generate detailed menu data."""
        menu_items = []
        for _ in range(num_items):
            item = {
                'item_id': f'ITEM_{random.randint(1000, 9999)}',
                'name': self._generate_dish_name(),
                'category': self._generate_menu_category(),
                'description': self._generate_dish_description(),
                'ingredients': self._generate_ingredients(),
                'nutritional_info': self._generate_nutritional_info(),
                'pricing': self._generate_pricing(),
                'preparation': self._generate_preparation_info(),
                'allergen_info': self._generate_allergen_info(),
                'dietary_info': self._generate_dietary_info(),
                'seasonal_availability': self._generate_seasonal_availability(),
                'popularity_metrics': self._generate_popularity_metrics(),
                'cost_analysis': self._generate_cost_analysis(),
                'wine_pairings': self._generate_wine_pairings(),
                'modification_options': self._generate_modification_options()
            }
            menu_items.append(item)
        return pd.DataFrame(menu_items)

    def _generate_dish_name(self) -> str:
        """Generate a realistic dish name."""
        cuisines = ['Italian', 'French', 'Japanese', 'Indian', 'Mexican', 'Thai', 'Mediterranean']
        cooking_methods = ['Grilled', 'Roasted', 'Braised', 'Seared', 'Poached', 'Steamed', 'Fried']
        ingredients = ['Salmon', 'Chicken', 'Beef', 'Lamb', 'Tofu', 'Shrimp', 'Mushroom']
        styles = ['Classic', 'Modern', 'Fusion', 'Traditional', 'Contemporary']
        
        return f"{random.choice(cooking_methods)} {random.choice(ingredients)} {random.choice(styles)} {random.choice(cuisines)} Style"

    def _generate_menu_category(self) -> Dict[str, Any]:
        """Generate menu category information."""
        return {
            'main_category': random.choice(['Appetizers', 'Main Courses', 'Desserts', 'Beverages', 'Sides']),
            'sub_category': random.choice(['Seafood', 'Meat', 'Vegetarian', 'Vegan', 'Gluten-Free']),
            'course_type': random.choice(['Starter', 'Main', 'Dessert', 'Drink']),
            'spice_level': random.choice(['Mild', 'Medium', 'Spicy']),
            'preparation_time': random.randint(10, 45)  # minutes
        }

    def _generate_dish_description(self) -> str:
        """Generate a detailed dish description."""
        cooking_methods = ['slow-cooked', 'pan-seared', 'roasted', 'grilled', 'braised']
        ingredients = ['fresh herbs', 'local vegetables', 'premium cuts', 'artisanal ingredients']
        flavors = ['rich', 'aromatic', 'savory', 'sweet', 'spicy']
        textures = ['crispy', 'tender', 'creamy', 'crunchy']
        
        return f"A {random.choice(cooking_methods)} dish featuring {random.choice(ingredients)}, " \
               f"with {random.choice(flavors)} flavors and {random.choice(textures)} textures."

    def _generate_ingredients(self) -> Dict[str, Any]:
        """Generate detailed ingredient information."""
        return {
            'main_ingredients': random.sample([
                'chicken breast', 'salmon fillet', 'beef tenderloin', 'tofu',
                'mushrooms', 'pasta', 'rice', 'quinoa'
            ], random.randint(1, 3)),
            'vegetables': random.sample([
                'carrots', 'broccoli', 'spinach', 'bell peppers',
                'zucchini', 'asparagus', 'kale'
            ], random.randint(1, 4)),
            'herbs_and_spices': random.sample([
                'basil', 'thyme', 'rosemary', 'oregano',
                'cumin', 'coriander', 'paprika'
            ], random.randint(2, 5)),
            'sauces_and_condiments': random.sample([
                'olive oil', 'balsamic vinegar', 'soy sauce',
                'teriyaki sauce', 'hot sauce'
            ], random.randint(1, 3)),
            'garnishes': random.sample([
                'fresh herbs', 'microgreens', 'citrus zest',
                'toasted nuts', 'seeds'
            ], random.randint(0, 2))
        }

    def _generate_nutritional_info(self) -> Dict[str, Any]:
        """Generate nutritional information."""
        return {
            'calories': random.randint(200, 800),
            'protein': random.randint(10, 40),
            'carbohydrates': random.randint(20, 60),
            'fat': random.randint(5, 30),
            'fiber': random.randint(2, 15),
            'sugar': random.randint(0, 20),
            'sodium': random.randint(100, 1000),
            'allergens': random.sample([
                'dairy', 'nuts', 'shellfish', 'wheat',
                'soy', 'eggs', 'fish'
            ], random.randint(0, 3))
        }

    def _generate_pricing(self) -> Dict[str, Any]:
        """Generate pricing information."""
        base_price = random.uniform(10, 50)
        return {
            'base_price': round(base_price, 2),
            'happy_hour_price': round(base_price * 0.8, 2),
            'weekend_price': round(base_price * 1.1, 2),
            'special_occasion_price': round(base_price * 1.2, 2),
            'portion_sizes': {
                'small': round(base_price * 0.7, 2),
                'regular': base_price,
                'large': round(base_price * 1.3, 2)
            }
        }

    def _generate_preparation_info(self) -> Dict[str, Any]:
        """Generate preparation information."""
        return {
            'prep_time': random.randint(5, 30),  # minutes
            'cook_time': random.randint(10, 45),  # minutes
            'total_time': random.randint(15, 75),  # minutes
            'difficulty_level': random.choice(['Easy', 'Medium', 'Hard']),
            'cooking_methods': random.sample([
                'grilling', 'roasting', 'sautéing', 'steaming',
                'braising', 'frying', 'baking'
            ], random.randint(1, 3)),
            'equipment_needed': random.sample([
                'oven', 'stovetop', 'grill', 'food processor',
                'mixer', 'blender'
            ], random.randint(1, 3))
        }

    def _generate_allergen_info(self) -> Dict[str, Any]:
        """Generate allergen information."""
        return {
            'contains': random.sample([
                'dairy', 'nuts', 'shellfish', 'wheat',
                'soy', 'eggs', 'fish'
            ], random.randint(0, 3)),
            'may_contain': random.sample([
                'dairy', 'nuts', 'shellfish', 'wheat',
                'soy', 'eggs', 'fish'
            ], random.randint(0, 2)),
            'preparation_notes': random.sample([
                'prepared in shared kitchen',
                'separate preparation area available',
                'cross-contamination possible'
            ], random.randint(0, 2))
        }

    def _generate_dietary_info(self) -> Dict[str, Any]:
        """Generate dietary information."""
        return {
            'suitable_for': random.sample([
                'vegetarian', 'vegan', 'gluten-free',
                'dairy-free', 'halal', 'kosher'
            ], random.randint(0, 3)),
            'modifications_available': random.sample([
                'can be made vegetarian',
                'can be made vegan',
                'can be made gluten-free',
                'can be made dairy-free'
            ], random.randint(0, 3)),
            'nutritional_highlights': random.sample([
                'high protein',
                'low carb',
                'high fiber',
                'low fat',
                'rich in vitamins'
            ], random.randint(0, 2))
        }

    def _generate_seasonal_availability(self) -> Dict[str, Any]:
        """Generate seasonal availability information."""
        return {
            'available_seasons': random.sample([
                'spring', 'summer', 'fall', 'winter'
            ], random.randint(1, 4)),
            'peak_season': random.choice(['spring', 'summer', 'fall', 'winter']),
            'limited_time': random.random() < 0.3,
            'seasonal_ingredients': random.sample([
                'asparagus', 'berries', 'pumpkin', 'citrus',
                'truffles', 'morel mushrooms'
            ], random.randint(0, 3))
        }

    def _generate_popularity_metrics(self) -> Dict[str, Any]:
        """Generate popularity metrics."""
        return {
            'order_frequency': random.uniform(0.1, 1.0),
            'customer_rating': random.uniform(3.5, 5.0),
            'review_count': random.randint(10, 1000),
            'trending_score': random.uniform(0, 1.0),
            'repeat_order_rate': random.uniform(0.1, 0.8),
            'social_media_mentions': random.randint(0, 1000)
        }

    def _generate_cost_analysis(self) -> Dict[str, Any]:
        """Generate cost analysis information."""
        return {
            'ingredient_cost': random.uniform(5, 25),
            'labor_cost': random.uniform(2, 10),
            'overhead_cost': random.uniform(1, 5),
            'total_cost': random.uniform(8, 40),
            'profit_margin': random.uniform(0.2, 0.4),
            'cost_trends': {
                'ingredient_cost_trend': random.uniform(-0.1, 0.1),
                'labor_cost_trend': random.uniform(-0.05, 0.05),
                'overhead_cost_trend': random.uniform(-0.05, 0.05)
            }
        }

    def _generate_wine_pairings(self) -> Dict[str, Any]:
        """Generate wine pairing information."""
        return {
            'recommended_wines': random.sample([
                'Chardonnay', 'Cabernet Sauvignon', 'Pinot Noir',
                'Sauvignon Blanc', 'Merlot', 'Syrah'
            ], random.randint(1, 3)),
            'pairing_notes': random.sample([
                'complements the rich flavors',
                'balances the spice',
                'enhances the umami notes',
                'contrasts with the sweetness'
            ], random.randint(1, 2)),
            'alternative_beverages': random.sample([
                'craft beer',
                'cocktail',
                'non-alcoholic option'
            ], random.randint(0, 2))
        }

    def _generate_modification_options(self) -> Dict[str, Any]:
        """Generate modification options."""
        return {
            'available_modifications': random.sample([
                'extra spicy',
                'mild',
                'no onions',
                'extra sauce',
                'no sauce',
                'gluten-free option',
                'dairy-free option'
            ], random.randint(0, 5)),
            'substitution_options': random.sample([
                'vegetable substitution',
                'protein substitution',
                'grain substitution',
                'sauce substitution'
            ], random.randint(0, 3)),
            'customization_notes': random.sample([
                'can be made spicier',
                'can be made milder',
                'can be made vegetarian',
                'can be made vegan'
            ], random.randint(0, 2))
        }

    def generate_inventory_data(self, num_items=200):
        """Generate detailed inventory data."""
        inventory_items = []
        for _ in range(num_items):
            item = {
                'item_id': f'INV_{random.randint(1000, 9999)}',
                'name': self._generate_inventory_item_name(),
                'category': self._generate_inventory_category(),
                'supplier_info': self._generate_supplier_info(),
                'stock_info': self._generate_stock_info(),
                'pricing_info': self._generate_inventory_pricing(),
                'quality_metrics': self._generate_quality_metrics(),
                'storage_info': self._generate_storage_info(),
                'usage_metrics': self._generate_usage_metrics(),
                'sustainability_info': self._generate_sustainability_info()
            }
            inventory_items.append(item)
        return pd.DataFrame(inventory_items)

    def _generate_inventory_item_name(self) -> str:
        """Generate an inventory item name."""
        categories = ['Produce', 'Meat', 'Seafood', 'Dairy', 'Dry Goods', 'Beverages']
        items = {
            'Produce': ['Organic', 'Local', 'Seasonal'],
            'Meat': ['Premium', 'Grass-fed', 'Free-range'],
            'Seafood': ['Wild-caught', 'Sustainable', 'Fresh'],
            'Dairy': ['Artisanal', 'Organic', 'Local'],
            'Dry Goods': ['Premium', 'Organic', 'Specialty'],
            'Beverages': ['Craft', 'Premium', 'Artisanal']
        }
        
        category = random.choice(categories)
        prefix = random.choice(items[category])
        item_type = random.choice([
            'Vegetables', 'Fruits', 'Beef', 'Chicken', 'Fish',
            'Cheese', 'Milk', 'Flour', 'Spices', 'Wine', 'Beer'
        ])
        
        return f"{prefix} {item_type}"

    def _generate_inventory_category(self) -> Dict[str, Any]:
        """Generate inventory category information."""
        return {
            'main_category': random.choice([
                'Produce', 'Meat', 'Seafood', 'Dairy',
                'Dry Goods', 'Beverages', 'Frozen'
            ]),
            'sub_category': random.choice([
                'Fresh', 'Frozen', 'Canned', 'Dried',
                'Refrigerated', 'Ambient'
            ]),
            'storage_type': random.choice([
                'Dry Storage', 'Refrigeration', 'Freezer',
                'Wine Cellar', 'Produce Storage'
            ]),
            'shelf_life': random.randint(1, 365)  # days
        }

    def _generate_supplier_info(self) -> Dict[str, Any]:
        """Generate supplier information."""
        return {
            'supplier_name': f"Supplier_{random.randint(100, 999)}",
            'supplier_type': random.choice([
                'Local Farm', 'Wholesale Distributor',
                'Specialty Importer', 'Local Producer'
            ]),
            'delivery_frequency': random.choice([
                'Daily', 'Weekly', 'Bi-weekly', 'Monthly'
            ]),
            'lead_time': random.randint(1, 14),  # days
            'quality_rating': random.uniform(3.0, 5.0),
            'reliability_score': random.uniform(0.7, 1.0)
        }

    def _generate_stock_info(self) -> Dict[str, Any]:
        """Generate stock information."""
        return {
            'current_stock': random.randint(0, 1000),
            'minimum_stock': random.randint(10, 100),
            'maximum_stock': random.randint(200, 2000),
            'reorder_point': random.randint(20, 200),
            'reorder_quantity': random.randint(50, 500),
            'stock_turnover_rate': random.uniform(1, 12),  # times per year
            'stock_value': random.uniform(100, 10000)
        }

    def _generate_inventory_pricing(self) -> Dict[str, Any]:
        """Generate inventory pricing information."""
        base_price = random.uniform(1, 100)
        return {
            'unit_price': round(base_price, 2),
            'bulk_price': round(base_price * 0.8, 2),
            'wholesale_price': round(base_price * 0.7, 2),
            'price_history': {
                'last_month': round(base_price * random.uniform(0.9, 1.1), 2),
                'last_quarter': round(base_price * random.uniform(0.8, 1.2), 2),
                'last_year': round(base_price * random.uniform(0.7, 1.3), 2)
            },
            'price_trend': random.choice(['increasing', 'stable', 'decreasing'])
        }

    def _generate_quality_metrics(self) -> Dict[str, Any]:
        """Generate quality metrics."""
        return {
            'quality_grade': random.choice(['A', 'B', 'C']),
            'freshness_score': random.uniform(0.7, 1.0),
            'consistency_score': random.uniform(0.7, 1.0),
            'quality_issues': random.sample([
                'occasional inconsistency',
                'variable freshness',
                'packaging issues',
                'delivery delays'
            ], random.randint(0, 2)),
            'quality_improvements': random.sample([
                'improved packaging',
                'better storage conditions',
                'faster delivery',
                'quality control measures'
            ], random.randint(0, 2))
        }

    def _generate_storage_info(self) -> Dict[str, Any]:
        """Generate storage information."""
        return {
            'storage_location': random.choice([
                'Dry Storage', 'Refrigeration', 'Freezer',
                'Wine Cellar', 'Produce Storage'
            ]),
            'temperature_range': {
                'min': random.uniform(-20, 10),
                'max': random.uniform(0, 25)
            },
            'humidity_range': {
                'min': random.uniform(30, 50),
                'max': random.uniform(60, 80)
            },
            'storage_requirements': random.sample([
                'keep dry',
                'refrigerate after opening',
                'store in cool place',
                'protect from light'
            ], random.randint(0, 2))
        }

    def _generate_usage_metrics(self) -> Dict[str, Any]:
        """Generate usage metrics."""
        return {
            'daily_usage': random.uniform(1, 100),
            'weekly_usage': random.uniform(10, 500),
            'monthly_usage': random.uniform(50, 2000),
            'usage_trend': random.choice(['increasing', 'stable', 'decreasing']),
            'waste_rate': random.uniform(0.01, 0.1),
            'efficiency_score': random.uniform(0.7, 1.0)
        }

    def _generate_sustainability_info(self) -> Dict[str, Any]:
        """Generate sustainability information."""
        return {
            'sustainability_rating': random.uniform(0, 1.0),
            'environmental_impact': random.choice(['low', 'medium', 'high']),
            'sustainable_practices': random.sample([
                'local sourcing',
                'organic production',
                'recyclable packaging',
                'reduced waste'
            ], random.randint(0, 3)),
            'certifications': random.sample([
                'organic',
                'fair trade',
                'sustainable seafood',
                'rainforest alliance'
            ], random.randint(0, 2))
        } 

    def generate_staff_data(self, num_staff=50):
        """Generate detailed staff data."""
        staff_members = []
        for _ in range(num_staff):
            staff = {
                'staff_id': f'STAFF_{random.randint(1000, 9999)}',
                'personal_info': self._generate_personal_info(),
                'employment_info': self._generate_employment_info(),
                'skills_and_certifications': self._generate_skills_and_certifications(),
                'performance_metrics': self._generate_performance_metrics(),
                'training_history': self._generate_training_history(),
                'schedule_preferences': self._generate_schedule_preferences(),
                'compensation_info': self._generate_compensation_info(),
                'career_development': self._generate_career_development()
            }
            staff_members.append(staff)
        return pd.DataFrame(staff_members)

    def _generate_personal_info(self) -> Dict[str, Any]:
        """Generate personal information."""
        return {
            'first_name': random.choice([
                'John', 'Jane', 'Michael', 'Sarah', 'David',
                'Emily', 'James', 'Emma', 'William', 'Olivia'
            ]),
            'last_name': random.choice([
                'Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
                'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'
            ]),
            'contact_info': {
                'email': f"staff_{random.randint(100, 999)}@quantai.restaurant",
                'phone': f"+1{random.randint(2000000000, 9999999999)}",
                'emergency_contact': f"+1{random.randint(2000000000, 9999999999)}"
            },
            'address': {
                'street': f"{random.randint(1, 999)} Main St",
                'city': random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']),
                'state': random.choice(['NY', 'CA', 'IL', 'TX', 'AZ']),
                'zip_code': f"{random.randint(10000, 99999)}"
            },
            'personal_details': {
                'date_of_birth': f"{random.randint(1980, 2000)}-{random.randint(1, 12)}-{random.randint(1, 28)}",
                'gender': random.choice(['Male', 'Female', 'Non-binary']),
                'marital_status': random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
            }
        }

    def _generate_employment_info(self) -> Dict[str, Any]:
        """Generate employment information."""
        return {
            'position': random.choice([
                'Chef', 'Sous Chef', 'Line Cook', 'Prep Cook',
                'Server', 'Bartender', 'Host', 'Manager',
                'Dishwasher', 'Busser'
            ]),
            'department': random.choice([
                'Kitchen', 'Front of House', 'Bar', 'Management',
                'Support Staff'
            ]),
            'employment_status': random.choice([
                'Full-time', 'Part-time', 'Seasonal', 'Temporary'
            ]),
            'hire_date': f"{random.randint(2018, 2023)}-{random.randint(1, 12)}-{random.randint(1, 28)}",
            'employment_history': [
                {
                    'position': random.choice([
                        'Chef', 'Sous Chef', 'Line Cook', 'Prep Cook',
                        'Server', 'Bartender', 'Host', 'Manager'
                    ]),
                    'start_date': f"{random.randint(2015, 2022)}-{random.randint(1, 12)}-{random.randint(1, 28)}",
                    'end_date': f"{random.randint(2016, 2023)}-{random.randint(1, 12)}-{random.randint(1, 28)}"
                }
                for _ in range(random.randint(0, 2))
            ]
        }

    def _generate_skills_and_certifications(self) -> Dict[str, Any]:
        """Generate skills and certifications information."""
        return {
            'technical_skills': random.sample([
                'Food Safety', 'Knife Skills', 'Menu Planning',
                'Inventory Management', 'Customer Service',
                'Wine Knowledge', 'Mixology', 'Point of Sale'
            ], random.randint(3, 6)),
            'soft_skills': random.sample([
                'Communication', 'Teamwork', 'Leadership',
                'Problem Solving', 'Time Management',
                'Conflict Resolution', 'Adaptability'
            ], random.randint(3, 6)),
            'certifications': random.sample([
                'ServSafe', 'Food Handler', 'Wine Certification',
                'Mixology Certification', 'First Aid',
                'CPR', 'Allergen Awareness'
            ], random.randint(1, 4)),
            'languages': random.sample([
                'English', 'Spanish', 'French', 'Mandarin',
                'Japanese', 'Italian', 'German'
            ], random.randint(1, 3)),
            'specializations': random.sample([
                'Fine Dining', 'Pastry', 'Wine Service',
                'Craft Cocktails', 'Vegetarian Cuisine',
                'Seafood', 'Grilling'
            ], random.randint(1, 3))
        }

    def _generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate performance metrics."""
        return {
            'attendance': {
                'attendance_rate': random.uniform(0.85, 1.0),
                'tardiness_rate': random.uniform(0, 0.1),
                'absenteeism_rate': random.uniform(0, 0.1)
            },
            'quality_metrics': {
                'customer_satisfaction': random.uniform(3.5, 5.0),
                'error_rate': random.uniform(0, 0.1),
                'efficiency_score': random.uniform(0.7, 1.0)
            },
            'productivity': {
                'tasks_completed': random.uniform(0.8, 1.0),
                'time_efficiency': random.uniform(0.7, 1.0),
                'quality_consistency': random.uniform(0.7, 1.0)
            },
            'teamwork': {
                'collaboration_score': random.uniform(0.7, 1.0),
                'communication_score': random.uniform(0.7, 1.0),
                'leadership_score': random.uniform(0.7, 1.0)
            }
        }

    def _generate_training_history(self) -> Dict[str, Any]:
        """Generate training history."""
        return {
            'completed_trainings': [
                {
                    'training_name': random.choice([
                        'Food Safety', 'Customer Service',
                        'Wine Service', 'Kitchen Safety',
                        'Point of Sale', 'Team Building'
                    ]),
                    'completion_date': f"{random.randint(2020, 2023)}-{random.randint(1, 12)}-{random.randint(1, 28)}",
                    'certification': random.random() < 0.7,
                    'score': random.uniform(70, 100)
                }
                for _ in range(random.randint(2, 6))
            ],
            'upcoming_trainings': [
                {
                    'training_name': random.choice([
                        'Advanced Wine Service', 'Leadership Development',
                        'Advanced Food Safety', 'Customer Experience',
                        'Inventory Management', 'Team Leadership'
                    ]),
                    'scheduled_date': f"{random.randint(2023, 2024)}-{random.randint(1, 12)}-{random.randint(1, 28)}",
                    'status': random.choice(['Scheduled', 'Pending', 'In Progress'])
                }
                for _ in range(random.randint(0, 2))
            ],
            'training_needs': random.sample([
                'Advanced Wine Knowledge',
                'Leadership Skills',
                'Inventory Management',
                'Customer Service Excellence',
                'Team Management'
            ], random.randint(0, 2))
        }

    def _generate_schedule_preferences(self) -> Dict[str, Any]:
        """Generate schedule preferences."""
        return {
            'preferred_shifts': random.sample([
                'Morning', 'Afternoon', 'Evening', 'Night'
            ], random.randint(1, 4)),
            'preferred_days': random.sample([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday'
            ], random.randint(3, 7)),
            'availability': {
                'weekday_morning': random.random() < 0.8,
                'weekday_evening': random.random() < 0.8,
                'weekend_morning': random.random() < 0.6,
                'weekend_evening': random.random() < 0.6
            },
            'time_off_preferences': {
                'vacation_preference': random.choice(['Summer', 'Winter', 'Spring', 'Fall']),
                'holiday_preference': random.sample([
                    'Christmas', 'Thanksgiving', 'New Year',
                    'Easter', 'Independence Day'
                ], random.randint(0, 2))
            }
        }

    def _generate_compensation_info(self) -> Dict[str, Any]:
        """Generate compensation information."""
        return {
            'base_salary': random.uniform(20000, 80000),
            'hourly_rate': random.uniform(10, 30),
            'overtime_rate': random.uniform(1.5, 2.0),
            'benefits': random.sample([
                'Health Insurance', 'Dental Insurance',
                'Vision Insurance', '401(k)', 'Paid Time Off',
                'Employee Discount', 'Professional Development'
            ], random.randint(3, 6)),
            'bonuses': {
                'performance_bonus': random.uniform(0, 0.1),
                'holiday_bonus': random.uniform(0, 0.05),
                'referral_bonus': random.uniform(0, 0.02)
            },
            'compensation_history': [
                {
                    'date': f"{random.randint(2020, 2023)}-{random.randint(1, 12)}-{random.randint(1, 28)}",
                    'change_type': random.choice(['Raise', 'Bonus', 'Adjustment']),
                    'amount': random.uniform(1000, 5000)
                }
                for _ in range(random.randint(0, 2))
            ]
        }

    def _generate_career_development(self) -> Dict[str, Any]:
        """Generate career development information."""
        return {
            'career_goals': random.sample([
                'Become Head Chef',
                'Open Own Restaurant',
                'Become Restaurant Manager',
                'Specialize in Pastry',
                'Develop Wine Program'
            ], random.randint(1, 3)),
            'development_plan': {
                'short_term_goals': random.sample([
                    'Complete Advanced Certification',
                    'Learn New Cuisine',
                    'Improve Leadership Skills',
                    'Master Wine Service'
                ], random.randint(1, 2)),
                'long_term_goals': random.sample([
                    'Become Executive Chef',
                    'Start Own Business',
                    'Become Regional Manager',
                    'Develop Training Program'
                ], random.randint(1, 2))
            },
            'mentorship': {
                'is_mentor': random.random() < 0.3,
                'is_mentee': random.random() < 0.3,
                'mentorship_goals': random.sample([
                    'Leadership Development',
                    'Skill Enhancement',
                    'Career Guidance',
                    'Industry Knowledge'
                ], random.randint(0, 2))
            },
            'succession_planning': {
                'potential_roles': random.sample([
                    'Head Chef',
                    'Restaurant Manager',
                    'Wine Director',
                    'Training Manager'
                ], random.randint(0, 2)),
                'readiness_level': random.choice(['High', 'Medium', 'Low']),
                'development_needs': random.sample([
                    'Leadership Training',
                    'Business Management',
                    'Financial Acumen',
                    'Strategic Planning'
                ], random.randint(0, 2))
            }
        }

    def generate_schedule_data(self, num_schedules=100):
        """Generate detailed schedule data."""
        schedules = []
        for _ in range(num_schedules):
            schedule = {
                'schedule_id': f'SCH_{random.randint(1000, 9999)}',
                'staff_info': self._generate_staff_schedule_info(),
                'shift_info': self._generate_shift_info(),
                'coverage_info': self._generate_coverage_info(),
                'compliance_info': self._generate_compliance_info(),
                'efficiency_metrics': self._generate_efficiency_metrics()
            }
            schedules.append(schedule)
        return pd.DataFrame(schedules)

    def _generate_staff_schedule_info(self) -> Dict[str, Any]:
        """Generate staff schedule information."""
        return {
            'staff_id': f'STAFF_{random.randint(1000, 9999)}',
            'position': random.choice([
                'Chef', 'Sous Chef', 'Line Cook', 'Prep Cook',
                'Server', 'Bartender', 'Host', 'Manager'
            ]),
            'department': random.choice([
                'Kitchen', 'Front of House', 'Bar', 'Management'
            ]),
            'employment_type': random.choice([
                'Full-time', 'Part-time', 'Seasonal'
            ])
        }

    def _generate_shift_info(self) -> Dict[str, Any]:
        """Generate shift information."""
        return {
            'shift_type': random.choice([
                'Morning', 'Afternoon', 'Evening', 'Night'
            ]),
            'start_time': f"{random.randint(0, 23):02d}:{random.choice(['00', '15', '30', '45'])}",
            'end_time': f"{random.randint(0, 23):02d}:{random.choice(['00', '15', '30', '45'])}",
            'break_duration': random.randint(15, 60),  # minutes
            'break_times': [
                f"{random.randint(0, 23):02d}:{random.choice(['00', '15', '30', '45'])}"
                for _ in range(random.randint(1, 2))
            ],
            'overtime_hours': random.uniform(0, 5),
            'shift_status': random.choice([
                'Scheduled', 'Completed', 'Cancelled', 'Modified'
            ])
        }

    def _generate_coverage_info(self) -> Dict[str, Any]:
        """Generate coverage information."""
        return {
            'required_staff': random.randint(5, 20),
            'scheduled_staff': random.randint(5, 20),
            'coverage_gaps': random.randint(0, 3),
            'department_coverage': {
                'kitchen': random.uniform(0.8, 1.0),
                'front_of_house': random.uniform(0.8, 1.0),
                'bar': random.uniform(0.8, 1.0),
                'management': random.uniform(0.8, 1.0)
            },
            'skill_coverage': {
                'food_preparation': random.uniform(0.8, 1.0),
                'customer_service': random.uniform(0.8, 1.0),
                'bartending': random.uniform(0.8, 1.0),
                'management': random.uniform(0.8, 1.0)
            }
        }

    def _generate_compliance_info(self) -> Dict[str, Any]:
        """Generate compliance information."""
        return {
            'labor_laws': {
                'maximum_hours': random.uniform(0.8, 1.0),
                'break_compliance': random.uniform(0.8, 1.0),
                'overtime_compliance': random.uniform(0.8, 1.0)
            },
            'safety_regulations': {
                'staff_ratios': random.uniform(0.8, 1.0),
                'certification_coverage': random.uniform(0.8, 1.0),
                'training_compliance': random.uniform(0.8, 1.0)
            },
            'scheduling_policies': {
                'fair_distribution': random.uniform(0.8, 1.0),
                'preference_accommodation': random.uniform(0.8, 1.0),
                'rotation_compliance': random.uniform(0.8, 1.0)
            }
        }

    def _generate_efficiency_metrics(self) -> Dict[str, Any]:
        """Generate efficiency metrics."""
        return {
            'labor_efficiency': {
                'staff_utilization': random.uniform(0.7, 1.0),
                'productivity_score': random.uniform(0.7, 1.0),
                'cost_efficiency': random.uniform(0.7, 1.0)
            },
            'service_efficiency': {
                'response_time': random.uniform(0.7, 1.0),
                'customer_satisfaction': random.uniform(0.7, 1.0),
                'service_quality': random.uniform(0.7, 1.0)
            },
            'operational_efficiency': {
                'resource_allocation': random.uniform(0.7, 1.0),
                'workflow_optimization': random.uniform(0.7, 1.0),
                'process_efficiency': random.uniform(0.7, 1.0)
            }
        }

    def generate_financial_data(self, num_records=1000):
        """Generate detailed financial data."""
        financial_records = []
        for _ in range(num_records):
            record = {
                'record_id': f'FIN_{random.randint(1000, 9999)}',
                'revenue_data': self._generate_revenue_data(),
                'cost_data': self._generate_cost_data(),
                'profitability_metrics': self._generate_profitability_metrics(),
                'cash_flow_data': self._generate_cash_flow_data(),
                'financial_ratios': self._generate_financial_ratios(),
                'budget_data': self._generate_budget_data(),
                'tax_data': self._generate_tax_data(),
                'investment_data': self._generate_investment_data()
            }
            financial_records.append(record)
        return pd.DataFrame(financial_records)

    def _generate_revenue_data(self) -> Dict[str, Any]:
        """Generate revenue data."""
        return {
            'total_revenue': random.uniform(5000, 50000),
            'revenue_breakdown': {
                'food_sales': random.uniform(3000, 30000),
                'beverage_sales': random.uniform(1000, 15000),
                'catering_sales': random.uniform(500, 5000),
                'merchandise_sales': random.uniform(100, 1000),
                'other_sales': random.uniform(100, 1000)
            },
            'revenue_trends': {
                'daily_growth': random.uniform(-0.1, 0.1),
                'weekly_growth': random.uniform(-0.05, 0.05),
                'monthly_growth': random.uniform(-0.02, 0.02)
            },
            'revenue_by_segment': {
                'dine_in': random.uniform(0.6, 0.8),
                'takeout': random.uniform(0.1, 0.2),
                'delivery': random.uniform(0.1, 0.2)
            }
        }

    def _generate_cost_data(self) -> Dict[str, Any]:
        """Generate cost data."""
        return {
            'total_costs': random.uniform(3000, 30000),
            'cost_breakdown': {
                'food_cost': random.uniform(1000, 10000),
                'labor_cost': random.uniform(1000, 8000),
                'overhead_cost': random.uniform(500, 5000),
                'utilities': random.uniform(200, 2000),
                'rent': random.uniform(500, 5000),
                'marketing': random.uniform(200, 2000),
                'maintenance': random.uniform(100, 1000),
                'other_costs': random.uniform(100, 1000)
            },
            'cost_trends': {
                'daily_change': random.uniform(-0.1, 0.1),
                'weekly_change': random.uniform(-0.05, 0.05),
                'monthly_change': random.uniform(-0.02, 0.02)
            },
            'cost_control_metrics': {
                'food_cost_percentage': random.uniform(0.25, 0.35),
                'labor_cost_percentage': random.uniform(0.25, 0.35),
                'overhead_percentage': random.uniform(0.15, 0.25)
            }
        }

    def _generate_profitability_metrics(self) -> Dict[str, Any]:
        """Generate profitability metrics."""
        return {
            'gross_profit': random.uniform(2000, 20000),
            'operating_profit': random.uniform(1000, 10000),
            'net_profit': random.uniform(500, 5000),
            'profit_margins': {
                'gross_margin': random.uniform(0.6, 0.7),
                'operating_margin': random.uniform(0.1, 0.2),
                'net_margin': random.uniform(0.05, 0.15)
            },
            'profitability_trends': {
                'daily_change': random.uniform(-0.1, 0.1),
                'weekly_change': random.uniform(-0.05, 0.05),
                'monthly_change': random.uniform(-0.02, 0.02)
            }
        }

    def _generate_cash_flow_data(self) -> Dict[str, Any]:
        """Generate cash flow data."""
        return {
            'operating_cash_flow': random.uniform(1000, 10000),
            'investing_cash_flow': random.uniform(-5000, 5000),
            'financing_cash_flow': random.uniform(-2000, 2000),
            'net_cash_flow': random.uniform(-1000, 1000),
            'cash_flow_metrics': {
                'cash_conversion_cycle': random.uniform(10, 30),
                'working_capital': random.uniform(5000, 50000),
                'free_cash_flow': random.uniform(500, 5000)
            },
            'cash_flow_trends': {
                'daily_change': random.uniform(-0.1, 0.1),
                'weekly_change': random.uniform(-0.05, 0.05),
                'monthly_change': random.uniform(-0.02, 0.02)
            }
        }

    def _generate_financial_ratios(self) -> Dict[str, Any]:
        """Generate financial ratios."""
        return {
            'liquidity_ratios': {
                'current_ratio': random.uniform(1.0, 2.0),
                'quick_ratio': random.uniform(0.5, 1.5),
                'cash_ratio': random.uniform(0.2, 0.8)
            },
            'efficiency_ratios': {
                'inventory_turnover': random.uniform(5, 15),
                'receivables_turnover': random.uniform(10, 30),
                'asset_turnover': random.uniform(1.0, 3.0)
            },
            'profitability_ratios': {
                'return_on_assets': random.uniform(0.05, 0.15),
                'return_on_equity': random.uniform(0.1, 0.2),
                'return_on_investment': random.uniform(0.1, 0.25)
            },
            'leverage_ratios': {
                'debt_ratio': random.uniform(0.3, 0.7),
                'debt_to_equity': random.uniform(0.5, 1.5),
                'interest_coverage': random.uniform(2.0, 5.0)
            }
        }

    def _generate_budget_data(self) -> Dict[str, Any]:
        """Generate budget data."""
        return {
            'total_budget': random.uniform(10000, 100000),
            'budget_allocation': {
                'operations': random.uniform(0.4, 0.6),
                'marketing': random.uniform(0.1, 0.2),
                'maintenance': random.uniform(0.1, 0.2),
                'development': random.uniform(0.1, 0.2),
                'contingency': random.uniform(0.05, 0.1)
            },
            'budget_performance': {
                'variance': random.uniform(-0.1, 0.1),
                'efficiency': random.uniform(0.8, 1.0),
                'utilization': random.uniform(0.7, 0.9)
            },
            'budget_trends': {
                'monthly_change': random.uniform(-0.05, 0.05),
                'quarterly_change': random.uniform(-0.1, 0.1),
                'annual_change': random.uniform(-0.15, 0.15)
            }
        }

    def _generate_tax_data(self) -> Dict[str, Any]:
        """Generate tax data."""
        return {
            'total_tax_liability': random.uniform(1000, 10000),
            'tax_breakdown': {
                'income_tax': random.uniform(500, 5000),
                'sales_tax': random.uniform(300, 3000),
                'property_tax': random.uniform(100, 1000),
                'payroll_tax': random.uniform(100, 1000)
            },
            'tax_compliance': {
                'filing_status': random.choice(['On Time', 'Late', 'Extension']),
                'audit_status': random.choice(['No Audit', 'Under Review', 'Audited']),
                'compliance_score': random.uniform(0.8, 1.0)
            },
            'tax_planning': {
                'deductions': random.uniform(1000, 10000),
                'credits': random.uniform(100, 1000),
                'tax_savings': random.uniform(500, 5000)
            }
        }

    def _generate_investment_data(self) -> Dict[str, Any]:
        """Generate investment data."""
        return {
            'total_investment': random.uniform(50000, 500000),
            'investment_breakdown': {
                'equipment': random.uniform(20000, 200000),
                'facility': random.uniform(20000, 200000),
                'technology': random.uniform(5000, 50000),
                'training': random.uniform(2000, 20000),
                'marketing': random.uniform(1000, 10000)
            },
            'investment_returns': {
                'roi': random.uniform(0.1, 0.3),
                'payback_period': random.uniform(1, 5),
                'npv': random.uniform(10000, 100000)
            },
            'investment_risks': {
                'market_risk': random.uniform(0.1, 0.3),
                'operational_risk': random.uniform(0.1, 0.3),
                'financial_risk': random.uniform(0.1, 0.3)
            }
        }

    def generate_analytics_data(self, num_records=1000):
        """Generate detailed analytics data."""
        analytics_records = []
        for _ in range(num_records):
            record = {
                'record_id': f'ANAL_{random.randint(1000, 9999)}',
                'customer_analytics': self._generate_customer_analytics(),
                'operational_analytics': self._generate_operational_analytics(),
                'financial_analytics': self._generate_financial_analytics(),
                'marketing_analytics': self._generate_marketing_analytics(),
                'performance_analytics': self._generate_performance_analytics(),
                'predictive_analytics': self._generate_predictive_analytics()
            }
            analytics_records.append(record)
        return pd.DataFrame(analytics_records)

    def _generate_customer_analytics(self) -> Dict[str, Any]:
        """Generate customer analytics data."""
        return {
            'customer_segments': {
                'loyal_customers': random.uniform(0.2, 0.4),
                'regular_customers': random.uniform(0.3, 0.5),
                'occasional_customers': random.uniform(0.2, 0.4),
                'new_customers': random.uniform(0.1, 0.3)
            },
            'customer_behavior': {
                'average_spend': random.uniform(30, 100),
                'visit_frequency': random.uniform(1, 10),
                'peak_visiting_times': random.sample([
                    'lunch', 'dinner', 'weekend', 'holiday'
                ], random.randint(1, 4)),
                'preferred_items': random.sample([
                    'appetizers', 'main_courses', 'desserts',
                    'beverages', 'special_items'
                ], random.randint(1, 5))
            },
            'customer_satisfaction': {
                'overall_rating': random.uniform(3.5, 5.0),
                'service_rating': random.uniform(3.5, 5.0),
                'food_rating': random.uniform(3.5, 5.0),
                'ambiance_rating': random.uniform(3.5, 5.0)
            },
            'customer_loyalty': {
                'retention_rate': random.uniform(0.6, 0.9),
                'churn_rate': random.uniform(0.1, 0.4),
                'loyalty_program_participation': random.uniform(0.3, 0.7)
            }
        }

    def _generate_operational_analytics(self) -> Dict[str, Any]:
        """Generate operational analytics data."""
        return {
            'efficiency_metrics': {
                'table_turnover_rate': random.uniform(2.0, 3.0),
                'average_service_time': random.uniform(45, 90),
                'kitchen_efficiency': random.uniform(0.7, 0.9),
                'staff_productivity': random.uniform(0.7, 0.9)
            },
            'quality_metrics': {
                'food_quality_score': random.uniform(0.8, 1.0),
                'service_quality_score': random.uniform(0.8, 1.0),
                'cleanliness_score': random.uniform(0.8, 1.0),
                'consistency_score': random.uniform(0.8, 1.0)
            },
            'resource_utilization': {
                'staff_utilization': random.uniform(0.7, 0.9),
                'equipment_utilization': random.uniform(0.7, 0.9),
                'space_utilization': random.uniform(0.7, 0.9),
                'inventory_utilization': random.uniform(0.7, 0.9)
            },
            'operational_costs': {
                'labor_cost_per_cover': random.uniform(5, 15),
                'food_cost_per_cover': random.uniform(10, 25),
                'overhead_per_cover': random.uniform(5, 15),
                'total_cost_per_cover': random.uniform(20, 55)
            }
        }

    def _generate_financial_analytics(self) -> Dict[str, Any]:
        """Generate financial analytics data."""
        return {
            'revenue_analytics': {
                'revenue_per_cover': random.uniform(30, 80),
                'revenue_per_table': random.uniform(100, 300),
                'revenue_per_hour': random.uniform(200, 1000),
                'revenue_growth_rate': random.uniform(-0.1, 0.2)
            },
            'cost_analytics': {
                'cost_per_cover': random.uniform(20, 50),
                'cost_per_table': random.uniform(70, 200),
                'cost_per_hour': random.uniform(150, 700),
                'cost_growth_rate': random.uniform(-0.1, 0.2)
            },
            'profitability_analytics': {
                'profit_per_cover': random.uniform(5, 30),
                'profit_per_table': random.uniform(20, 150),
                'profit_per_hour': random.uniform(50, 300),
                'profit_growth_rate': random.uniform(-0.1, 0.2)
            },
            'financial_health': {
                'current_ratio': random.uniform(1.0, 2.0),
                'quick_ratio': random.uniform(0.5, 1.5),
                'debt_to_equity': random.uniform(0.5, 1.5),
                'return_on_investment': random.uniform(0.1, 0.3)
            }
        }

    def _generate_marketing_analytics(self) -> Dict[str, Any]:
        """Generate marketing analytics data."""
        return {
            'campaign_performance': {
                'email_campaign_roi': random.uniform(1.0, 5.0),
                'social_media_engagement': random.uniform(0.1, 0.5),
                'promotional_effectiveness': random.uniform(0.5, 0.9),
                'customer_acquisition_cost': random.uniform(20, 100)
            },
            'channel_effectiveness': {
                'direct_traffic': random.uniform(0.2, 0.4),
                'referral_traffic': random.uniform(0.1, 0.3),
                'social_media_traffic': random.uniform(0.1, 0.3),
                'search_traffic': random.uniform(0.1, 0.3)
            },
            'customer_acquisition': {
                'new_customer_rate': random.uniform(0.1, 0.3),
                'conversion_rate': random.uniform(0.1, 0.3),
                'acquisition_cost': random.uniform(20, 100),
                'lifetime_value': random.uniform(100, 1000)
            },
            'brand_metrics': {
                'brand_awareness': random.uniform(0.3, 0.7),
                'brand_reputation': random.uniform(0.5, 0.9),
                'brand_loyalty': random.uniform(0.3, 0.7),
                'brand_equity': random.uniform(0.3, 0.7)
            }
        }

    def _generate_performance_analytics(self) -> Dict[str, Any]:
        """Generate performance analytics data."""
        return {
            'staff_performance': {
                'productivity_score': random.uniform(0.7, 1.0),
                'quality_score': random.uniform(0.7, 1.0),
                'efficiency_score': random.uniform(0.7, 1.0),
                'customer_satisfaction_score': random.uniform(0.7, 1.0)
            },
            'operational_performance': {
                'service_speed': random.uniform(0.7, 1.0),
                'order_accuracy': random.uniform(0.7, 1.0),
                'cleanliness_score': random.uniform(0.7, 1.0),
                'safety_compliance': random.uniform(0.7, 1.0)
            },
            'financial_performance': {
                'revenue_per_employee': random.uniform(50000, 150000),
                'profit_per_employee': random.uniform(10000, 30000),
                'cost_efficiency': random.uniform(0.7, 1.0),
                'return_on_assets': random.uniform(0.1, 0.3)
            },
            'overall_performance': {
                'customer_satisfaction': random.uniform(0.7, 1.0),
                'operational_efficiency': random.uniform(0.7, 1.0),
                'financial_health': random.uniform(0.7, 1.0),
                'employee_satisfaction': random.uniform(0.7, 1.0)
            }
        }

    def _generate_predictive_analytics(self) -> Dict[str, Any]:
        """Generate predictive analytics data."""
        return {
            'demand_forecasting': {
                'daily_demand': random.uniform(50, 200),
                'weekly_demand': random.uniform(300, 1000),
                'monthly_demand': random.uniform(1000, 5000),
                'seasonal_trends': random.sample([
                    'increasing', 'decreasing', 'stable', 'cyclical'
                ], random.randint(1, 2))
            },
            'revenue_forecasting': {
                'daily_revenue': random.uniform(2000, 10000),
                'weekly_revenue': random.uniform(10000, 50000),
                'monthly_revenue': random.uniform(50000, 200000),
                'growth_rate': random.uniform(-0.1, 0.2)
            },
            'customer_behavior_prediction': {
                'visit_probability': random.uniform(0.1, 0.9),
                'spending_prediction': random.uniform(30, 100),
                'churn_probability': random.uniform(0.1, 0.5),
                'loyalty_prediction': random.uniform(0.3, 0.9)
            },
            'operational_predictions': {
                'staffing_needs': random.uniform(5, 20),
                'inventory_requirements': random.uniform(1000, 5000),
                'equipment_maintenance': random.uniform(0.1, 0.5),
                'resource_allocation': random.uniform(0.5, 0.9)
            }
        }

    def _initialize_beverage_program(self) -> Dict[str, Any]:
        """Initialize the restaurant's beverage program."""
        return {
            'wine_program': {
                'by_the_glass': {
                    'red': [
                        {'name': 'House Red', 'varietal': 'Cabernet Sauvignon', 'region': 'California', 'price': 12},
                        {'name': 'Pinot Noir', 'varietal': 'Pinot Noir', 'region': 'Oregon', 'price': 14},
                        {'name': 'Malbec', 'varietal': 'Malbec', 'region': 'Argentina', 'price': 13}
                    ],
                    'white': [
                        {'name': 'House White', 'varietal': 'Chardonnay', 'region': 'California', 'price': 11},
                        {'name': 'Sauvignon Blanc', 'varietal': 'Sauvignon Blanc', 'region': 'New Zealand', 'price': 13},
                        {'name': 'Pinot Grigio', 'varietal': 'Pinot Grigio', 'region': 'Italy', 'price': 12}
                    ],
                    'sparkling': [
                        {'name': 'Prosecco', 'varietal': 'Glera', 'region': 'Italy', 'price': 12},
                        {'name': 'Cava', 'varietal': 'Macabeo', 'region': 'Spain', 'price': 13}
                    ]
                },
                'by_the_bottle': {
                    'red': [
                        {'name': 'Cabernet Sauvignon', 'varietal': 'Cabernet Sauvignon', 'region': 'Napa Valley', 'price': 65},
                        {'name': 'Pinot Noir', 'varietal': 'Pinot Noir', 'region': 'Willamette Valley', 'price': 58},
                        {'name': 'Malbec', 'varietal': 'Malbec', 'region': 'Mendoza', 'price': 45}
                    ],
                    'white': [
                        {'name': 'Chardonnay', 'varietal': 'Chardonnay', 'region': 'Sonoma', 'price': 55},
                        {'name': 'Sauvignon Blanc', 'varietal': 'Sauvignon Blanc', 'region': 'Marlborough', 'price': 48},
                        {'name': 'Pinot Grigio', 'varietal': 'Pinot Grigio', 'region': 'Veneto', 'price': 42}
                    ],
                    'sparkling': [
                        {'name': 'Champagne', 'varietal': 'Chardonnay/Pinot Noir', 'region': 'Champagne', 'price': 85},
                        {'name': 'Prosecco', 'varietal': 'Glera', 'region': 'Veneto', 'price': 45}
                    ]
                }
            },
            'craft_cocktails': {
                'signature': [
                    {'name': 'QuantAI Old Fashioned', 'ingredients': ['Bourbon', 'Bitters', 'Orange'], 'price': 14},
                    {'name': 'Restaurant Mule', 'ingredients': ['Vodka', 'Ginger Beer', 'Lime'], 'price': 13},
                    {'name': 'Chef\'s Manhattan', 'ingredients': ['Rye', 'Sweet Vermouth', 'Bitters'], 'price': 15}
                ],
                'seasonal': [
                    {'name': 'Spring Garden', 'ingredients': ['Gin', 'Elderflower', 'Cucumber'], 'price': 14},
                    {'name': 'Summer Breeze', 'ingredients': ['Vodka', 'Peach', 'Prosecco'], 'price': 13},
                    {'name': 'Autumn Spice', 'ingredients': ['Bourbon', 'Apple', 'Cinnamon'], 'price': 14},
                    {'name': 'Winter Warmer', 'ingredients': ['Whiskey', 'Hot Chocolate', 'Marshmallow'], 'price': 15}
                ]
            },
            'beer_program': {
                'draft': [
                    {'name': 'House Lager', 'style': 'Lager', 'brewery': 'Local Brewery', 'price': 7},
                    {'name': 'IPA', 'style': 'India Pale Ale', 'brewery': 'Craft Brewery', 'price': 8},
                    {'name': 'Wheat Beer', 'style': 'Hefeweizen', 'brewery': 'German Brewery', 'price': 8}
                ],
                'bottle': [
                    {'name': 'Belgian Ale', 'style': 'Belgian Strong Ale', 'brewery': 'Belgian Brewery', 'price': 9},
                    {'name': 'Stout', 'style': 'Imperial Stout', 'brewery': 'Craft Brewery', 'price': 10},
                    {'name': 'Saison', 'style': 'Farmhouse Ale', 'brewery': 'Local Brewery', 'price': 9}
                ]
            },
            'non_alcoholic': {
                'mocktails': [
                    {'name': 'Virgin Mojito', 'ingredients': ['Mint', 'Lime', 'Soda'], 'price': 8},
                    {'name': 'Berry Spritz', 'ingredients': ['Mixed Berries', 'Sparkling Water'], 'price': 8},
                    {'name': 'Cucumber Cooler', 'ingredients': ['Cucumber', 'Lime', 'Mint'], 'price': 8}
                ],
                'specialty_drinks': [
                    {'name': 'House Lemonade', 'ingredients': ['Fresh Lemon', 'Honey', 'Mint'], 'price': 7},
                    {'name': 'Iced Tea', 'ingredients': ['Black Tea', 'Peach', 'Lemon'], 'price': 6},
                    {'name': 'Sparkling Water', 'ingredients': ['Mineral Water', 'Citrus'], 'price': 5}
                ]
            },
            'coffee_program': {
                'espresso_based': [
                    {'name': 'Espresso', 'price': 3},
                    {'name': 'Cappuccino', 'price': 4},
                    {'name': 'Latte', 'price': 4.5},
                    {'name': 'Mocha', 'price': 5}
                ],
                'specialty_coffee': [
                    {'name': 'Pour Over', 'price': 5},
                    {'name': 'Cold Brew', 'price': 5},
                    {'name': 'French Press', 'price': 4}
                ]
            },
            'tea_program': {
                'hot_tea': [
                    {'name': 'Earl Grey', 'type': 'Black Tea', 'price': 4},
                    {'name': 'Green Tea', 'type': 'Green Tea', 'price': 4},
                    {'name': 'Chamomile', 'type': 'Herbal Tea', 'price': 4}
                ],
                'iced_tea': [
                    {'name': 'Black Tea', 'price': 4},
                    {'name': 'Green Tea', 'price': 4},
                    {'name': 'Herbal Tea', 'price': 4}
                ]
            }
        }

    def _initialize_suppliers(self) -> Dict[str, Any]:
        """Initialize supplier information for the restaurant."""
        return {
            'produce_suppliers': [
                {
                    'name': 'Fresh Harvest Farms',
                    'type': 'Local Organic Produce',
                    'delivery_frequency': 'Daily',
                    'payment_terms': 'Net 30',
                    'quality_rating': 4.8,
                    'reliability_rating': 4.9,
                    'specialties': ['Organic Vegetables', 'Herbs', 'Microgreens']
                },
                {
                    'name': 'Valley View Produce',
                    'type': 'Regional Produce',
                    'delivery_frequency': 'Daily',
                    'payment_terms': 'Net 15',
                    'quality_rating': 4.6,
                    'reliability_rating': 4.7,
                    'specialties': ['Seasonal Fruits', 'Root Vegetables']
                }
            ],
            'meat_suppliers': [
                {
                    'name': 'Premium Meats Co.',
                    'type': 'Premium Meats',
                    'delivery_frequency': '3x Weekly',
                    'payment_terms': 'Net 30',
                    'quality_rating': 4.9,
                    'reliability_rating': 4.8,
                    'specialties': ['Wagyu Beef', 'Heritage Pork', 'Free-range Poultry']
                },
                {
                    'name': 'Local Butcher Shop',
                    'type': 'Local Meats',
                    'delivery_frequency': '2x Weekly',
                    'payment_terms': 'Net 15',
                    'quality_rating': 4.7,
                    'reliability_rating': 4.6,
                    'specialties': ['Local Beef', 'Lamb', 'Game Meats']
                }
            ],
            'seafood_suppliers': [
                {
                    'name': 'Ocean Fresh Seafood',
                    'type': 'Sustainable Seafood',
                    'delivery_frequency': 'Daily',
                    'payment_terms': 'Net 15',
                    'quality_rating': 4.8,
                    'reliability_rating': 4.7,
                    'specialties': ['Fresh Fish', 'Shellfish', 'Sushi-grade Fish']
                }
            ],
            'dairy_suppliers': [
                {
                    'name': 'Cream Valley Dairy',
                    'type': 'Artisanal Dairy',
                    'delivery_frequency': '3x Weekly',
                    'payment_terms': 'Net 15',
                    'quality_rating': 4.7,
                    'reliability_rating': 4.8,
                    'specialties': ['Artisanal Cheeses', 'Organic Milk', 'Cream']
                }
            ],
            'wine_suppliers': [
                {
                    'name': 'Vintage Wines',
                    'type': 'Fine Wines',
                    'delivery_frequency': 'Weekly',
                    'payment_terms': 'Net 30',
                    'quality_rating': 4.9,
                    'reliability_rating': 4.8,
                    'specialties': ['Premium Wines', 'Rare Vintages']
                },
                {
                    'name': 'Local Vineyards',
                    'type': 'Local Wines',
                    'delivery_frequency': 'Bi-weekly',
                    'payment_terms': 'Net 15',
                    'quality_rating': 4.6,
                    'reliability_rating': 4.7,
                    'specialties': ['Local Wines', 'Small Batch Productions']
                }
            ]
        }

    def _initialize_training_programs(self) -> Dict[str, Any]:
        """Initialize staff training programs."""
        return {
            'food_safety': {
                'servsafe_certification': {
                    'duration': '8 hours',
                    'frequency': 'Every 3 years',
                    'instructor': 'Certified ServSafe Instructor',
                    'topics': ['Food Safety', 'Sanitation', 'HACCP']
                },
                'allergen_awareness': {
                    'duration': '4 hours',
                    'frequency': 'Annually',
                    'instructor': 'Food Safety Manager',
                    'topics': ['Allergen Identification', 'Cross-contamination Prevention']
                }
            },
            'service_training': {
                'wine_service': {
                    'duration': '6 hours',
                    'frequency': 'Quarterly',
                    'instructor': 'Sommelier',
                    'topics': ['Wine Knowledge', 'Service Techniques', 'Food Pairing']
                },
                'customer_service': {
                    'duration': '4 hours',
                    'frequency': 'Monthly',
                    'instructor': 'Service Manager',
                    'topics': ['Guest Relations', 'Conflict Resolution', 'Upselling']
                }
            },
            'culinary_training': {
                'cooking_techniques': {
                    'duration': '8 hours',
                    'frequency': 'Monthly',
                    'instructor': 'Executive Chef',
                    'topics': ['Advanced Techniques', 'Menu Items', 'Quality Control']
                },
                'menu_knowledge': {
                    'duration': '4 hours',
                    'frequency': 'Weekly',
                    'instructor': 'Sous Chef',
                    'topics': ['Ingredients', 'Preparation Methods', 'Allergens']
                }
            }
        }

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Initialize the dataset generator
    generator = QuantAIRestaurantDatasetGenerator()
    
    try:
        # Generate all datasets
        print("Generating customer demographics...")
        customer_df = generator.generate_customer_demographics(num_customers=1000)
        customer_df.to_csv(os.path.join(data_dir, "quantai_restaurant_customers.csv"), index=False)
        
        print("Generating reservations...")
        reservation_df = generator.generate_reservations(customer_df, num_reservations=2000)
        reservation_df.to_csv(os.path.join(data_dir, "quantai_restaurant_reservations.csv"), index=False)
        
        print("Generating orders...")
        order_df = generator.generate_orders(customer_df, num_orders=5000)
        order_df.to_csv(os.path.join(data_dir, "quantai_restaurant_orders.csv"), index=False)
        
        print("Generating menu data...")
        menu_df = generator.generate_menu_data(num_items=100)
        menu_df.to_csv(os.path.join(data_dir, "quantai_restaurant_menu.csv"), index=False)
        
        print("Generating inventory data...")
        inventory_df = generator.generate_inventory_data(num_items=200)
        inventory_df.to_csv(os.path.join(data_dir, "quantai_restaurant_inventory.csv"), index=False)
        
        print("Generating staff data...")
        staff_df = generator.generate_staff_data(num_staff=50)
        staff_df.to_csv(os.path.join(data_dir, "quantai_restaurant_staff_schedule.csv"), index=False)
        
        print("Generating financial data...")
        financial_df = generator.generate_financial_data(num_records=1000)
        financial_df.to_csv(os.path.join(data_dir, "quantai_restaurant_financial.csv"), index=False)
        
        print("Generating analytics data...")
        analytics_df = generator.generate_analytics_data(num_records=1000)
        analytics_df.to_csv(os.path.join(data_dir, "quantai_restaurant_analytics.csv"), index=False)
        
        # Save restaurant infrastructure data
        print("Saving restaurant infrastructure data...")
        infrastructure_data = {
            'restaurant_name': generator.restaurant_name,
            'owner': generator.owner,
            'infrastructure': generator.infrastructure,
            'quality_metrics': generator.quality_metrics,
            'certifications': generator.certifications,
            'culinary_knowledge': generator.culinary_knowledge
        }
        with open(os.path.join(data_dir, "restaurant_infrastructure.json"), 'w') as f:
            json.dump(infrastructure_data, f, indent=4)
            
        print("\nDataset generation completed successfully!")
        print(f"All data files have been saved to the '{data_dir}' directory.")
        
    except Exception as e:
        print(f"Error generating dataset: {str(e)}")
        raise