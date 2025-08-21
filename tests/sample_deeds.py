"""
Sample deed texts for testing the parser.
"""

SAMPLE_DEEDS = {
    "simple_rectangle": """
    Beginning at a concrete monument found at the southeast corner of the intersection 
    of Main Street and Oak Avenue;
    
    THENCE N 0°00'00" E along the east line of Oak Avenue, a distance of 150.00 feet 
    to a point for corner;
    
    THENCE N 90°00'00" E, a distance of 200.00 feet to an iron rod set for corner;
    
    THENCE S 0°00'00" E, a distance of 150.00 feet to a point on the north line of Main Street;
    
    THENCE S 90°00'00" W along the north line of Main Street, a distance of 200.00 feet 
    to the POINT OF BEGINNING, containing 0.688 acres, more or less.
    """,
    
    "with_curves": """
    BEGINNING at a point on the north right-of-way line of State Highway 123;
    
    THENCE N 15°30'45" E, a distance of 125.50 feet to a point for corner;
    
    THENCE with a curve to the right having a radius of 285.00 feet, an arc length 
    of 42.15 feet, chord bearing N 19°45'30" E, chord length 42.08 feet to a point for corner;
    
    THENCE S 75°15'00" E, a distance of 200.00 feet to an iron rod found for corner;
    
    THENCE with a curve to the left having a radius of 150.00 feet, an arc length 
    of 78.54 feet, chord bearing S 60°00'00" E, chord length 76.54 feet to a point;
    
    THENCE S 15°30'45" W, a distance of 180.25 feet to a point on the north 
    right-of-way line of said State Highway 123;
    
    THENCE N 75°15'00" W along said north right-of-way line, a distance of 285.00 feet 
    to the POINT OF BEGINNING, containing 1.25 acres, more or less.
    """,
    
    "complex_property": """
    TRACT 1: Being a tract of land situated in the John Smith Survey, Abstract 123, 
    Harris County, Texas, and being more particularly described as follows:
    
    COMMENCING at a 5/8 inch iron rod found at the southwest corner of a called 
    10.00 acre tract described in Volume 456, Page 789 of the Harris County Deed Records;
    
    THENCE N 89°45'30" E along the south line of said 10.00 acre tract, a distance 
    of 35.50 feet to a 5/8 inch iron rod set for the POINT OF BEGINNING of the tract 
    herein described;
    
    THENCE continuing N 89°45'30" E along said south line, a distance of 150.75 feet 
    to a 5/8 inch iron rod set;
    
    THENCE N 15°30'00" E, a distance of 200.00 feet to a 5/8 inch iron rod set;
    
    THENCE with a curve to the left having a radius of 500.00 feet, a central angle 
    of 28°15'30", an arc length of 246.18 feet, and a chord bearing of N 01°22'15" E 
    with a chord length of 244.85 feet to a 5/8 inch iron rod set;
    
    THENCE N 12°45'00" W, a distance of 125.50 feet to a 5/8 inch iron rod set on 
    the south right-of-way line of County Road 789;
    
    THENCE S 77°15'00" W along said south right-of-way line, a distance of 285.75 feet 
    to a 5/8 inch iron rod found;
    
    THENCE with a curve to the right having a radius of 300.00 feet, an arc length 
    of 52.36 feet, chord bearing S 82°15'30" W, chord length 52.25 feet to a point;
    
    THENCE S 15°30'00" W, a distance of 475.25 feet to the POINT OF BEGINNING;
    
    CONTAINING 2.85 acres (124,146 square feet), more or less.
    """,
    
    "irregular_shape": """
    Beginning at a stone found, the same being the northeast corner of the 
    James Wilson 50-acre tract;
    
    THENCE S 45°15'30" E, 185.50 feet to a stake set;
    
    THENCE S 12°30'45" W, 95.75 feet to a post oak tree marked "X";
    
    THENCE with a curve to the right, radius = 200.00 feet, arc = 89.54 feet, 
    chord bearing S 25°45'15" W, chord = 88.75 feet to an iron pin set;
    
    THENCE N 78°30'00" W, 125.25 feet to a concrete monument set;
    
    THENCE N 35°15'45" W, 95.50 feet to a point on the south line of Oak Street;
    
    THENCE N 85°45'30" E along said south line, 165.75 feet to a point;
    
    THENCE N 15°30'15" E, 75.25 feet to the POINT OF BEGINNING;
    
    Containing 1.15 acres, more or less.
    """,
    
    "metes_and_bounds": """
    All that certain lot, tract or parcel of land lying and being situated in 
    the City of Austin, Travis County, Texas, and being Lot 15, Block C of the 
    Riverside Addition, according to the map or plat thereof recorded in Volume 
    25, Page 45 of the Map Records of Travis County, Texas, and being more 
    particularly described by metes and bounds as follows:
    
    BEGINNING at a 1/2 inch iron pipe found at the southeast corner of said Lot 15;
    
    THENCE N 01°15'30" W along the east line of said lot, 125.00 feet to a 
    1/2 inch iron pipe found at the northeast corner thereof;
    
    THENCE S 88°44'30" W along the north line of said lot, 75.00 feet to a 
    1/2 inch iron pipe found at the northwest corner thereof;
    
    THENCE S 01°15'30" E along the west line of said lot, 125.00 feet to a 
    1/2 inch iron pipe found at the southwest corner thereof;
    
    THENCE N 88°44'30" E along the south line of said lot, 75.00 feet to the 
    POINT OF BEGINNING;
    
    CONTAINING 9,375 square feet or 0.215 acres, more or less.
    """
}


def get_sample_deed(name: str) -> str:
    """Get a sample deed by name"""
    return SAMPLE_DEEDS.get(name, "")


def get_all_sample_names() -> list[str]:
    """Get all available sample deed names"""
    return list(SAMPLE_DEEDS.keys())


def get_sample_with_expected_results(name: str) -> dict:
    """Get sample deed with expected parsing results for testing"""
    
    expected_results = {
        "simple_rectangle": {
            "num_calls": 4,
            "perimeter": 700.0,  # 150 + 200 + 150 + 200
            "area": 30000.0,     # 150 * 200
            "closure_should_be_good": True
        },
        
        "with_curves": {
            "num_calls": 6,
            "has_curves": True,
            "closure_should_be_good": True
        },
        
        "complex_property": {
            "num_calls": 8,
            "has_curves": True,
            "area_approx": 124146.0,  # Given in deed
            "closure_should_be_good": True
        },
        
        "irregular_shape": {
            "num_calls": 7,
            "has_curves": True,
            "closure_should_be_good": True
        },
        
        "metes_and_bounds": {
            "num_calls": 4,
            "perimeter": 400.0,  # 125 + 75 + 125 + 75
            "area": 9375.0,      # Given in deed
            "closure_should_be_good": True
        }
    }
    
    return {
        "text": SAMPLE_DEEDS.get(name, ""),
        "expected": expected_results.get(name, {})
    }
