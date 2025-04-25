# Classroom Behavior Management System

## Overview
The Classroom Behavior Management System is an AI-powered application designed to help educators objectively monitor and analyze student behaviors in real-time using computer vision. By leveraging a fine-tuned YOLO v11s model, the system detects distractions such as phone usage, eating, looking around, and sleeping. It provides actionable analytics to improve classroom dynamics and long-term learning environments.

## Problem Statement
Traditional classroom management often lacks automated tools to objectively track undesirable student behaviors (e.g., sleeping, phone use). Teachers and schools struggle to quantify these patterns, making it difficult to implement data-driven improvements. Existing systems focus on basic actions like hand-raising and fail to address disruptive behaviors comprehensively.

## Key Features
- **Real-Time Behavior Detection**:  
  Identifies phone usage, eating, sleeping, and distracted behaviors via live camera feed.
- **Smart Alert System**:  
  - Immediate alerts for phone use and distraction.  
  - 10-second countdown alerts for eating/sleeping to avoid repeated triggers.  
- **Analytics & Reporting**:  
  - Tracks behavior frequency and stores data in a database.  
  - Displays historical alerts with timestamps and countdown analytics.  
- **Teacher Dashboard**:  
  - Live camera monitoring.  
  - Access to class-specific behavior reports.  
  - Filter data by date/time, class, or behavior type.  

## Installation
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/xkbang/BehaviourManagementSystem
   cd BehaviourManagementSystem
