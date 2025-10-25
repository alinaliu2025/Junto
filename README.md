# Junto - OSU Club Directory

A modern web application for discovering and connecting with clubs and organizations at Oregon State University.

## Features

- **Club Discovery**: Browse and search through all OSU clubs by category
- **Club Profiles**: Detailed club pages with meeting times, descriptions, and social media links
- **Member Profiles**: LinkedIn-style profiles for club members
- **Category Filtering**: Filter clubs by Greek Life, Professional, Cultural, Sports, Academic, and Special Interest
- **Search Functionality**: Find clubs by name or description
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern UI**: Built with shadcn/ui components for a polished experience

## Tech Stack

- **Frontend**: React 18 with TypeScript
- **UI Components**: shadcn/ui with Radix UI primitives
- **Build Tool**: Vite for fast development and building
- **Styling**: Tailwind CSS with custom OSU color scheme
- **Routing**: React Router for navigation
- **Icons**: Lucide React for consistent iconography
- **State Management**: TanStack Query for server state
- **Forms**: React Hook Form with Zod validation
- **Development**: ESLint for code quality

## Getting Started

1. **Install UI dependencies**:
   ```bash
   npm run install-ui
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

3. **Open your browser** and navigate to `http://localhost:8080`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run install-ui` - Install UI dependencies

## Project Structure

```
Junto/
├── UI/                           # Frontend React application
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── ui/              # shadcn/ui components
│   │   │   ├── ClubCard.tsx     # Club display component
│   │   │   ├── Navbar.tsx       # Navigation bar
│   │   │   ├── Sidebar.tsx      # Side navigation
│   │   │   ├── SearchBar.tsx    # Search functionality
│   │   │   └── FilterChips.tsx  # Category filters
│   │   ├── pages/               # Page components
│   │   │   ├── DiscoveryPage.tsx        # Main club discovery
│   │   │   ├── ClubProfilePage.tsx       # Individual club details
│   │   │   ├── MemberDashboardPage.tsx  # Member dashboard
│   │   │   ├── CreateClubPage.tsx       # Create new club
│   │   │   └── UserProfilePage.tsx      # User profiles
│   │   ├── data/                # Mock data and constants
│   │   ├── hooks/                # Custom React hooks
│   │   ├── lib/                  # Utility functions
│   │   ├── App.tsx               # Main app component with routing
│   │   ├── main.tsx              # App entry point
│   │   └── index.css             # Global styles and Tailwind imports
│   ├── package.json              # UI dependencies and scripts
│   ├── tailwind.config.ts        # Tailwind CSS configuration
│   ├── tsconfig.json             # TypeScript configuration
│   ├── vite.config.ts            # Vite build configuration
│   └── index.html                # HTML template
├── lib/                          # Backend API and utilities
├── prisma/                       # Database schema and migrations
├── supabase/                     # Supabase configuration
├── types/                        # TypeScript type definitions
├── package.json                  # Root project configuration
└── README.md                    # Project documentation
```

## Features Overview

### Discovery Page
- Modern search interface with real-time filtering
- Category-based filtering with chips
- Responsive grid layout for club cards
- Create new club functionality

### Club Profiles
- Detailed club information and descriptions
- Member management and roles
- Event creation and management
- Social media integration

### Member Dashboard
- Personal dashboard for club members
- Club membership management
- Event participation tracking
- Profile customization

### User Profiles
- LinkedIn-style member profiles
- Bio and interests management
- Club membership history
- Contact information

## Customization

The app uses OSU's official colors and modern design principles:
- **Primary Colors**: OSU Orange and complementary colors
- **Typography**: Inter font family for modern readability
- **Components**: shadcn/ui for consistent, accessible design
- **Responsive**: Mobile-first design approach

## Future Enhancements

- User authentication and profiles
- Real-time club updates
- Event management system
- Club application process
- Member directory with privacy controls
- Mobile app development
- Integration with OSU systems
- Advanced search and filtering
- Club analytics and insights

## Contributing

This is a student project for Oregon State University. Feel free to contribute improvements or report issues.

## License

This project is created for educational purposes at Oregon State University.