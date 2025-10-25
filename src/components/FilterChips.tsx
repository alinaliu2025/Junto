import { cn } from "@/lib/utils";

interface FilterChipsProps {
  categories: string[];
  selected: string;
  onSelect: (category: string) => void;
}

export function FilterChips({ categories, selected, onSelect }: FilterChipsProps) {
  return (
    <div className="flex gap-2 flex-wrap">
      {categories.map((category) => (
        <button
          key={category}
          onClick={() => onSelect(category)}
          className={cn(
            "px-4 py-2 rounded-full text-sm font-medium transition-smooth shadow-sm",
            selected === category
              ? "bg-primary text-primary-foreground"
              : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
          )}
        >
          {category}
        </button>
      ))}
    </div>
  );
}
