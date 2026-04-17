"""
College Service — loads JSON data and filters colleges based on student profile.
"""

import json
from typing import List, Dict, Any
from config import DATA_PATH


class CollegeService:
    """Loads and filters colleges from the JSON dataset."""

    def __init__(self):
        self.colleges: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        """Load colleges from JSON file."""
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.colleges = data.get("colleges", [])
            print(f"✅ Loaded {len(self.colleges)} colleges from dataset.")
        except FileNotFoundError:
            print("❌ colleges.json not found!")
            self.colleges = []
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in colleges.json: {e}")
            self.colleges = []

    # ------------------------------------------------------------------
    # Branch alias map – normalises user input to dataset values
    # ------------------------------------------------------------------
    BRANCH_ALIASES: Dict[str, List[str]] = {
        "ai/ml": ["ai & ml", "artificial intelligence", "machine learning", "ai and ml", "ai/ml"],
        "aiml": ["ai & ml", "artificial intelligence", "machine learning"],
        "cse": ["computer science", "computer engineering", "computer science & engineering"],
        "it": ["information technology", "it"],
        "ece": ["electronics & communication", "electronics and communication", "electronics"],
        "eee": ["electrical engineering", "electrical"],
        "mech": ["mechanical", "mechanical engineering"],
        "civil": ["civil", "civil engineering"],
        "ds": ["data science", "data science & analytics"],
    }

    # Cities that are part of Delhi NCR — include when user searches "delhi"
    NCR_CITIES = {"new delhi", "delhi", "noida", "greater noida", "gurugram", "gurgaon", "faridabad", "ghaziabad"}

    def _branch_matches(self, branch_input: str, college_branches: List[str]) -> int:
        """Return a match score (0 = no match, 8 = partial, 15 = exact)."""
        b_lower = branch_input.lower().strip()
        cb_lower = [b.lower() for b in college_branches]

        # Exact match
        if b_lower in cb_lower:
            return 15

        # Alias expansion
        aliases = self.BRANCH_ALIASES.get(b_lower, [])
        for alias in aliases:
            if any(alias in cb for cb in cb_lower):
                return 15

        # Partial substring match
        if any(b_lower in cb for cb in cb_lower):
            return 8

        # Reverse – check if any college branch token is in the input
        for cb in cb_lower:
            if cb in b_lower:
                return 8

        return 0

    def _location_matches(self, location_input: str, college: Dict[str, Any]) -> bool:
        """Return True if the college is in or near the requested location."""
        loc = location_input.lower().strip()
        city = college.get("city", "").lower()
        state = college.get("state", "").lower()

        # Direct match
        if loc in city or loc in state:
            return True

        # NCR expansion: treat all NCR cities as "Delhi"
        if loc in ("delhi", "new delhi") and city in self.NCR_CITIES:
            return True

        return False

    def filter_colleges(
        self,
        location: str,
        course: str,
        budget: int,
        marks: int,
        branch: str = None,
        category: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Filter colleges based on student profile.
        Priority: location > course > budget > branch.
        Returns a scored & sorted list.
        """
        results = []
        course_lower = course.lower().strip()

        for college in self.colleges:
            score = 0

            # --- Location check (with NCR expansion) ---
            if not self._location_matches(location, college):
                continue

            # --- Course check ---
            courses = [c.lower() for c in college.get("courses", [])]
            if course_lower not in courses:
                continue
            score += 30

            # --- Branch matching (with alias normalisation) ---
            if branch:
                branch_score = self._branch_matches(branch, college.get("branches", []))
                score += branch_score

            # --- Budget check ---
            fees = college.get("fees_per_year", 0)
            if fees > budget:
                continue

            # --- Marks vs cutoff ---
            cutoff = college.get("cutoff_rank", {}).get(category.lower(), 100000)
            if marks < 60 and cutoff < 50000:
                score -= 15
            elif marks > 85 and cutoff < 40000:
                score += 10

            # --- Placement bonus ---
            avg_lpa = college.get("placement_avg_lpa", 0)
            if avg_lpa >= 10:
                score += 10
            elif avg_lpa >= 7:
                score += 5

            # --- NAAC bonus ---
            naac = college.get("naac_grade", "")
            if "A++" in naac:
                score += 8
            elif "A+" in naac:
                score += 6
            elif naac == "A":
                score += 4

            if score > 0:
                results.append({**college, "_score": score})

        # Sort by score descending, then placement
        results.sort(key=lambda x: (x["_score"], x.get("placement_avg_lpa", 0)), reverse=True)
        return results[:15]

    def get_all_locations(self) -> List[str]:
        """Get unique cities and states."""
        cities = set()
        states = set()
        for c in self.colleges:
            cities.add(c.get("city", ""))
            states.add(c.get("state", ""))
        return sorted(cities | states)

    def get_all_states(self) -> List[str]:
        """Get unique states only."""
        return sorted({c.get("state", "") for c in self.colleges if c.get("state")})

    def get_all_courses(self) -> List[str]:
        """Get unique courses across all colleges."""
        courses = set()
        for c in self.colleges:
            courses.update(c.get("courses", []))
        return sorted(courses)

    def get_all_branches(self) -> List[str]:
        """Get unique branches across all colleges."""
        branches = set()
        for c in self.colleges:
            branches.update(c.get("branches", []))
        return sorted(branches)

    def get_all_types(self) -> List[str]:
        """Get unique college types (Government, Private, etc.)."""
        return sorted({c.get("type", "") for c in self.colleges if c.get("type")})

    def get_stats(self) -> Dict[str, Any]:
        """Return a rich summary of the dataset."""
        total = len(self.colleges)
        govt  = sum(1 for c in self.colleges if "government" in c.get("type", "").lower())
        pvt   = sum(1 for c in self.colleges if "private" in c.get("type", "").lower())
        with_hostel = sum(1 for c in self.colleges if c.get("hostel"))

        naac_counts: Dict[str, int] = {}
        for c in self.colleges:
            grade = c.get("naac_grade", "Unknown")
            naac_counts[grade] = naac_counts.get(grade, 0) + 1

        states = self.get_all_states()

        return {
            "total_colleges": total,
            "government_colleges": govt,
            "private_colleges": pvt,
            "colleges_with_hostel": with_hostel,
            "states_covered": len(states),
            "naac_distribution": naac_counts,
            "states": states,
        }
