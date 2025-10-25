import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Sidebar } from "@/components/Sidebar";
import { SearchBar } from "@/components/SearchBar";
import { FilterChips } from "@/components/FilterChips";
import { ClubCard } from "@/components/ClubCard";
import { Button } from "@/components/ui/button";
import { mockClubs, categories } from "@/data/mockData";

export default function DiscoveryPage() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");

  const filteredClubs = mockClubs.filter((club) => {
    const matchesSearch = club.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      club.shortDescription.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === "All" || club.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />
      
      <div className="flex flex-1">
        <Sidebar />

        <main className="flex-1 p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            {/* Search Bar and Create Button */}
            <div className="flex justify-center items-center gap-4">
              <SearchBar value={searchQuery} onChange={setSearchQuery} />
              <Button variant="join" onClick={() => navigate("/create-club")}>
                Create New Club
              </Button>
            </div>

            {/* Filter Chips */}
            <FilterChips
              categories={categories}
              selected={selectedCategory}
              onSelect={setSelectedCategory}
            />

            {/* Results Count */}
            <div className="text-sm text-muted-foreground">
              {filteredClubs.length} {filteredClubs.length === 1 ? "club" : "clubs"} found
            </div>

            {/* Club Grid */}
            {filteredClubs.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-muted-foreground">No clubs found matching your criteria</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredClubs.map((club) => (
                  <ClubCard key={club.id} club={club} />
                ))}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
